from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

import torch
import torch.nn as nn
from torch.nn import functional as F

class AnomalyRadDINO(AnomalyDINOModel):
    """
    AnomalyRadDINO is a variant of AnomalyDINO that incorporates RadDINO (only pre-trained on CXR images)
    This allows the model to capture more domain-specific features relevant to chest X-ray anomaly detection, 
    potentially improving performance on medical imaging tasks.
    """
    def __init__(
        self, 
        raddino_model,
        backbone_layer: str = "p-1",
        patch_size: int = 16,
        num_neighbours: int | list[int] = 1,
        coreset_subsampling: bool = True,
        sampling_ratio: float = 0.01) -> None:
        
        DynamicBufferMixin.__init__(self)
        nn.Module.__init__(self)
        
        if isinstance(num_neighbours, int):
            self.num_neighbours = [num_neighbours]
        else:
            self.num_neighbours = num_neighbours
        
        self.coreset_subsampling = coreset_subsampling
        self.sampling_ratio = sampling_ratio
        
        # RadDINO Settings
        self.feature_encoder = raddino_model
        self.feature_encoder.eval()
        self.patch_size = patch_size
        self.backbone_layer = backbone_layer
        
        # Memory bank and embedding storage
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.Tensor] = []
        
        # Anomaly map generator for visualization and scoring
        self.anomaly_map_generator = AnomalyMapGenerator()
        
    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract patch features from the input image tensor using the RadDINO-based feature encoder.
        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Extracted patch features of shape (B, N, D), 
            where N is the number of patches and D is the feature dimension.
        """
        
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        expected_n = (h // self.patch_size) * (w // self.patch_size)
        
        with torch.no_grad():
            features = self.feature_encoder(image_tensor)[self.backbone_layer] 
        
        # The TimmBackbone outputs features in shape (B, C, H', W') for ViT-based models, where H' and W' correspond to the grid size of patches.
        if features.dim() == 4:
            features = features.flatten(2).transpose(1, 2) # Reshape to (B, N, D) where N = H'*W' and D = C
            if features.shape[1] == expected_n + 1:
                return features[:, 1:, :] # Remove the CLS token if it exists, as we only want patch features for anomaly detection.
            elif features.shape[1] == expected_n:
                return features
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}. Expected (B, N, D) where N is either {expected_n} or {expected_n + 1}.")
            
        # If the feature encoder already outputs in (B, N, D) format, we can directly check the shape.
        elif features.dim() == 3:
            if features.shape[1] == expected_n + 1:
                return features[:, 1:, :] # Remove the CLS token if it exists, as we only want patch features for anomaly detection.
            elif features.shape[1] == expected_n:
                return features
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}. Expected (B, N, D) where N is either {expected_n} or {expected_n + 1}.")
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}. Expected either (B, C, H', W') or (B, N, D).")
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """ 
        Forward pass for both training and inference.

        In training mode:
            - Extracts normalized patch features.
            - Collects embeddings into the memory bank.

        In inference mode:
            - Computes distances between input features and the memory bank.
            - Performs kNN-based scoring and anomaly map generation.

        Args:
            input_tensor (torch.Tensor): Input batch of shape ``(B, 3, H, W)``.

        Returns:
            Union[torch.Tensor, InferenceBatch]:
                - In training: dummy scalar tensor (no loss backprop).
                - In inference: :class:`anomalib.data.InferenceBatch` containing:
                    * ``pred_score``: Image-level anomaly score ``(B, 1)``
                    * ``anomaly_map``: Pixel-level anomaly heatmap ``(B, 1, H, W)``
        """
        
        # set precision
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        
        # work out sizing
        b, _, h, w = input_tensor.shape
        patch_size = self.patch_size
        
        # Center crop input to dimensions divisible by patch size
        # This avoids introducing artificial content while maintaining spatial alignment
        crop_h = h % patch_size
        crop_w = w % patch_size
        pad_top = crop_h // 2
        pad_bottom = crop_h - pad_top
        pad_left = crop_w // 2
        pad_right = crop_w - pad_left
        
        cropped_h = h - crop_h
        cropped_w = w - crop_w
        
        if crop_h > 0 or crop_w > 0:
            input_tensor = input_tensor[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]
            
        grid_size = (
            cropped_h // patch_size,
            cropped_w // patch_size
        )
        
        device = input_tensor.device
        features = self.extract_features(input_tensor) # [B, N, D]
        
        # Don't mask background features during training, as we want the memory bank to learn the full distribution of normal features.
        masks = torch.ones(features.shape[:2], dtype=torch.bool, device=device) # [B, N]
        
        features = features[masks]  # [Q, D], where Q is the total number of valid patches across the batch (as we don't mask any during training, Q = B*N)
        features = F.normalize(features, p=2, dim=1)
        
        if self.training:
            self.embedding_store.append(features)
            return torch.tensor(0.0, device=device, requires_grad=True)  # Dummy loss for training
        
        # check bank isn't empty at inference
        if self.memory_bank.numel() == 0:
            msg = "Memory bank is empty. Run the model in training mode and call `fit()` before inference."
            raise RuntimeError(msg)
        
        # Ensure dtype consistency
        if features.dtype != self.memory_bank.dtype:
            features = features.to(self.memory_bank.dtype)
            
        # Inference
        # L2-normalized distances
        # memory_bank : [M, D], features : [Q, D]

        # Compute cosine distance using matrix multiplication
        # both features and memory_bank are already L2-normalized.
        # cdist is not for half precision, but matmul is.
        similarity = torch.matmul(features, self.memory_bank.T)  # [Q, M]
        dists = (torch.ones_like(similarity) - similarity).clamp(min=0.0, max=2.0)  # cosine distance ∈ [0, 2]
        
        k_values = self.num_neighbours
        results = []
        
        max_k = max(1, max(k_values)) # For example: k_values = [1, 3, 5, 10]
        
        # For each patch feature q, find the top-k nearest neighbors in the memory bank and sort by distance. 
        # This gives us the k smallest distances for each patch.
        topk_vals_all, _ = torch.topk(dists, k=max_k, dim=1, largest=False) # [Q, max_k] - the k nearest neighbor distances for each patch, sorted from smallest to largest.
        
        for k in k_values:
            k_clamped = max(1, min(k, topk_vals_all.shape[1]))
            
            # Select the top-k distances for each patch. If k=1, this will give us the single nearest neighbor distance. If k>1, we get the k nearest neighbor distances.
            topk_vals = topk_vals_all[:, :k_clamped] # [Q, k_clamped] - the k nearest neighbor distances for each patch, sorted from smallest to largest.
            
            # Mean over k neighbors if needed
            min_dists = topk_vals.mean(dim=1) if k_clamped > 1 else topk_vals.squeeze(1) # [Q] - the anomaly score for each patch, which is the mean distance to its k nearest neighbors in the memory bank (or just the single nearest neighbor distance if k=1).
            
            # Vectorized reconstruction
            distances_full = torch.zeros(
                (b, grid_size[0] * grid_size[1]),
                device=device,
                dtype=min_dists.dtype,
            )
            batch_idx, patch_idx = torch.nonzero(masks, as_tuple=True)
            distances_full[batch_idx, patch_idx] = min_dists # [B, N] - the anomaly score for each patch in the original batch/grid layout, with zeros for any masked-out patches (though in our case we don't mask any during training, so all patches have a score).
            
            scores_dict = {
                "top1": self.mean_top_k(distances_full, top_k=1),
                "top5": self.mean_top_k(distances_full, top_k=5),
                "top10": self.mean_top_k(distances_full, top_k=10),
                "top1_percent": self.mean_top_percentile(distances_full, percentile=0.99),
                "top5_percent": self.mean_top_percentile(distances_full, percentile=0.95),
                "top10_percent": self.mean_top_percentile(distances_full, percentile=0.90),
                "softmax_weighted": self.softmax_weighted_mean(distances_full, top_ratio=0.05, tau=0.1)
            }
            
            # Aggregate image-level anomaly scores
            # image_score = self.mean_top_percentile(distances_full, percentile=0.95) # [B, 1] - the overall anomaly score for each image in the batch, computed as the mean of the top 1% highest patch anomaly scores.

            # Generate final anomaly map
            anomaly_map = distances_full.view(b, 1, *grid_size)
            anomaly_map = self.anomaly_map_generator(anomaly_map, (cropped_h, cropped_w)) # [B, 1, H, W] - the pixel-level anomaly heatmap for each image, upsampled to the original image size (before cropping).
            if crop_h > 0 or crop_w > 0:
                anomaly_map = F.pad(anomaly_map, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

            results.append({
                "k": k,
                "pred_scores": scores_dict,
                "anomaly_map": anomaly_map
            })
            # results.append(InferenceBatch(pred_score=image_score, anomaly_map=anomaly_map))
        
        return results if len(results) > 1 else results[0]
    
    @staticmethod
    def mean_top_percentile(distances: torch.Tensor, percentile: float = 0.95) -> torch.Tensor:
        """
        Aggregate patch-level anomaly scores using percentile-based method.
        
        Args:
            distances (torch.Tensor): Patch-level distances [B, N_patches]
            percentile (float): Percentile threshold (default 95 = top 5%)
        
        Returns:
            torch.Tensor: Image-level anomaly scores [B, 1]
        """
        n = distances.shape[-1]
        num_top = max(int(n * (1 - percentile)), 1)
        topk_vals, _ = torch.topk(distances, num_top, dim=1, largest=True)
        return topk_vals.mean(dim=1, keepdim=True)
    
    @staticmethod
    def mean_top_k(distances: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        
        n = distances.shape[-1]
        
        k_clamped = min(top_k, n)
        
        topk_vals, _ = torch.topk(distances, k_clamped, dim=1, largest=True)
        
        return topk_vals.mean(dim=1, keepdim=True)
    
    
    @staticmethod
    def softmax_weighted_mean(distances: torch.Tensor, top_ratio: float = 0.05, tau: float = 0.1) -> torch.Tensor:
        """
        Aggregate patch-level anomaly scores using a softmax-weighted mean of the top-k distances.
        """
        n = distances.shape[-1]
        
        num_top = max(int(n * top_ratio), 1)
        
        topk_vals, _ = torch.topk(distances, num_top, dim=1, largest=True)
        
        # Giảm temperature tau (<1) để khuếch đại vùng cách biệt
        weights = F.softmax(topk_vals / tau, dim=1)
        
        weighted_score = torch.sum(topk_vals * weights, dim=1, keepdim=True)
        
        return weighted_score
    
    def fit(self, normal_num: int) -> None:
        """Finalize and optionally subsample the memory bank after training.

        Once all embeddings from normal training images have been collected,
        this method consolidates them into the memory bank and optionally
        performs coreset-based subsampling.
        
        Args:
            normal_num (int): The total number of normal training samples used to build the memory bank. 
            This is required for coreset subsampling to determine the target size of the memory bank.

        Raises:
            ValueError: If called before collecting any embeddings.
        """
        if len(self.embedding_store) == 0:
            err_str = "No embeddings collected. Run model in training mode first."
            raise ValueError(err_str)
        
        self.memory_bank = torch.vstack(self.embedding_store)
        self.embedding_store.clear()

        if self.sampling_ratio == 1.0:
            print("Sampling ratio is 1.0, skipping coreset subsampling.")
            return 
        elif self.sampling_ratio <= 0.0 or self.sampling_ratio > 1.0:
            raise ValueError(f"Invalid sampling ratio: {self.sampling_ratio}. Must be in (0, 1].")
        else:
            print(f"Performing coreset subsampling with ratio {self.sampling_ratio}...")
            
            if normal_num < 1000:
                print(f"Normal dataset size ({normal_num}) is small. Coreset subsampling is performed on GPU for efficiency.")
                self.memory_bank = self.memory_bank.cuda()  # Ensure memory bank is on GPU for subsampling
                sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=self.sampling_ratio)
                self.memory_bank = sampler.sample_coreset()
            else:
                print(f"Normal dataset size ({normal_num}) is large. Coreset subsampling is performed on CPU to avoid GPU memory issues.")
                self.memory_bank = self.memory_bank.cpu()  # Move to CPU for subsampling
                sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=self.sampling_ratio)
                self.memory_bank = sampler.sample_coreset()