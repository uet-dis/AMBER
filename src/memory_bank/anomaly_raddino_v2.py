import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.memory_bank.anomaly_raddino import AnomalyRadDINO
from src.memory_bank.utils import SELECTED_ANATOMIES

import torch
from torch.nn import functional as F
from anomalib.data import InferenceBatch
from anomalib.models.components import KCenterGreedy

class AnomalyRadDINOv2(AnomalyRadDINO):
    """
    Anatomical-Aware Memory Bank (AAMB) Version.
    This version accepts anatomical masks to selectively store normal patches,
    filtering out uninformative background patches (e.g., text, edges, empty space).
    
    Maintains separate memory banks for each anatomical region.
    """
    
    MAX_GPU_PATCHES = 2000000 # Adjust based on available GPU memory and feature dimension. This is a heuristic to avoid OOM errors during subsampling.
    COVERAGE_THRESHOLD = 0.3     # A patch is considered to belong to an anatomical region if at least 30% of its area is covered by the corresponding binary mask.
    
    def __init__(self, num_anatomies: int = 5, full_anomaly_map: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_anatomies = num_anatomies
        
        del self.memory_bank
        del self.embedding_store
        
        self.embedding_stores = {i: [] for i in range(self.num_anatomies)}
        
        for i in range(self.num_anatomies):
            self.register_buffer(f"memory_bank_{i}", torch.empty(0))
        
        # If True, the model will generate pixel-level anomaly maps 
        # Else, the model will only return patch-level anomaly maps   
        self.full_anomaly_map = full_anomaly_map
    
    
    def fit(self) -> None:
        """
        After accumulating embeddings during training, this method consolidates them into memory banks.
        There are several memory banks corresponding to different anatomical regions. 
        The method also applies optional subsampling to manage memory size, especially for large datasets.
        
        Raises:
            ValueError: If called before collecting any embeddings.
        """
        
        for c in range(self.num_anatomies):
            if len(self.embedding_stores[c]) == 0:
                raise ValueError(f"No embeddings collected for organ {SELECTED_ANATOMIES[c]}. Cannot create memory bank.")
               
            bank_c = torch.vstack(self.embedding_stores[c])
            self.embedding_stores[c].clear()

            if self.sampling_ratio < 1.0:
                
                num_patches = bank_c.shape[0]
                
                print(f"[Organ {SELECTED_ANATOMIES[c]}] Subsampling from {bank_c.shape[0]} patches...")
                
                if bank_c.shape[0] > self.MAX_GPU_PATCHES:
                    print(f"[Organ {SELECTED_ANATOMIES[c]}] Too many patches for GPU subsampling. Applying random sampling to reduce to {self.MAX_GPU_PATCHES} patches before K-Center Greedy.")
                    indices = torch.randperm(bank_c.shape[0])[:self.MAX_GPU_PATCHES]
                    bank_c = bank_c[indices]
                    
                    target_coreset_size = int(num_patches * self.sampling_ratio)
                    
                    adjusted_ratio = target_coreset_size / self.MAX_GPU_PATCHES    
                
                else:
                    adjusted_ratio = self.sampling_ratio
                
                bank_c = bank_c.cuda()
                sampler = KCenterGreedy(embedding=bank_c, sampling_ratio=adjusted_ratio)    
                    
                bank_c = sampler.sample_coreset()
            else:
                print(f"[Organ {SELECTED_ANATOMIES[c]}] No subsampling applied. Total patches: {bank_c.shape[0]}")
                    
            setattr(self, f"memory_bank_{c}", bank_c)
            print(f"[Organ {SELECTED_ANATOMIES[c]}] Sub-bank created with shape: {bank_c.shape}")
    
        
    def forward(self, input_tensor: torch.Tensor, anatomy_masks: torch.Tensor = None, active_anatomies_idx: list[int] = None) -> torch.Tensor | InferenceBatch:
        
        input_tensor = input_tensor.type(getattr(self, f"memory_bank_0").dtype)
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
            if anatomy_masks is not None:
                anatomy_masks = anatomy_masks[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]
                
        grid_size = (cropped_h // patch_size, cropped_w // patch_size)
        
        device = input_tensor.device
        
        features = self.extract_features(input_tensor) # [B, N, D]
        
        features_norm = F.normalize(features, p=2, dim=2) # [B, N, D]
        
        # Create patch-level masks for each anatomical region. If anatomy_masks is None, fallback to treating all patches as valid.
        if anatomy_masks is not None:
            # As patch size is 16x16, a patch is considered to belong to an anatomical region if at least a pre-defined threshold of its area is covered by the corresponding binary mask.
            patch_masks = F.avg_pool2d(anatomy_masks.float(), kernel_size=patch_size, stride=patch_size) >= self.COVERAGE_THRESHOLD # [B, C, H//P, W//P]
        else:
            raise ValueError("anatomy_masks is required to generate patch-level masks for anatomical regions.")

        if self.training:
                
            for c in range(self.num_anatomies):
                masks_c = patch_masks[:, c, :, :].reshape(b, -1) # [B, H // P * W // P]
                feats_c = features_norm[masks_c]                 # [Q, D], Q = total number of patches belonging to organ c
                
                if feats_c.numel() > 0:
                    self.embedding_stores[c].append(feats_c)
                    
            return torch.tensor(0.0, device=device, requires_grad=True)

        # INFERENCE PHASE
        # Each sub-bank corresponds to an anatomical region. 
        # We compute the anomaly score for each patch based on its distance to the nearest neighbors in the corresponding sub-bank(s).
        # Sub-bank c has the shape: [M_c, D], where M_c is the number of normal patches for sub-bank corresponding to c-th organ
        
        k_values = self.num_neighbours
        max_k = max(1, max(k_values))
        
        # Initialize a tensor to store the maximum distance for each patch across all relevant sub-banks. This will be used to compute the final anomaly scores.
        distances_k = {k: torch.zeros((b, grid_size[0] * grid_size[1]), device=device, dtype=features_norm.dtype) for k in k_values}
        
        anatomies_to_process = active_anatomies_idx if active_anatomies_idx is not None else range(self.num_anatomies)
        
        for c in anatomies_to_process:
            
            bank_c = getattr(self, f"memory_bank_{c}") # [M_c, D]
            
            if bank_c.numel() == 0:
                raise ValueError(f"Memory bank for organ {SELECTED_ANATOMIES[c]} is empty. Cannot compute anomaly scores for this organ.")
                
            if features_norm.dtype != bank_c.dtype:
                features_norm = features_norm.to(bank_c.dtype)
            if bank_c.device != device:
                bank_c = bank_c.to(device)
                

            masks_c = patch_masks[:, c, :, :].reshape(b, -1) # [B, N], N = total number of patches in the image
            feats_c = features_norm[masks_c]                 # [Q, D], Q = total number of patches belonging to organ c, D is hidden dimension of features
            
            if feats_c.numel() == 0:
                continue
                
            # Calculate cosine similarity and convert to distance. Distance is in [0, 2], where 0 means identical and 2 means opposite.
            similarity = torch.matmul(feats_c, bank_c.T)     # [Q, M_c]
            dists_c = (torch.ones_like(similarity) - similarity).clamp(min=0.0, max=2.0)
            
            # # Return top K nearest neighbors for each patch feature comparing with the corresponding sub-bank.
            topk_vals_all, _ = torch.topk(dists_c, k=max_k, dim=1, largest=False) # [Q, max_k]
            
            # Return the coordinates of the patches that belong to the current organ. 
            # Assume batch_idx = [0, 0, ...] and patch_idx = [0, 1, 2, ...], it means that at the 0-the patch of 0-th image belongs to the current organ, 
            # at the 1-the patch of 0-th image belongs to the current organ, and so on.
            # When combining with the min_dists below, we could recover the original shape of the input batch and assign the anomaly scores to the correct locations in the anomaly map.
            batch_idx, patch_idx = torch.nonzero(masks_c, as_tuple=True)
            
            for k in k_values:
                
                k_clamped = max(1, min(k, topk_vals_all.shape[1]))
                
                topk_vals = topk_vals_all[:, :k_clamped]
                
                min_dists = topk_vals.mean(dim=1) if k_clamped > 1 else topk_vals.squeeze(1) # [Q]
                
                # If a patch belongs to multiple organs, we take the maximum distance (most anomalous) across all relevant sub-banks to ensure we don't miss any potential anomalies.
                current_scores = distances_k[k][batch_idx, patch_idx]
                distances_k[k][batch_idx, patch_idx] = torch.max(current_scores, min_dists)
        
        # After processing all anatomical regions, we have a final anomaly score for each patch based on the maximum distance to the nearest neighbors in the relevant sub-banks.
        # We can then generate the anomaly map and compute the overall image-level anomaly scores based on these patch-level distances.
        results = []
        for k in k_values:
            distances_full = distances_k[k]
            
            # Compute various image-level anomaly scores based on the patch-level distances. These scores can be used for evaluation or thresholding.
            scores_dict = {
                "top1": self.mean_top_k(distances_full, top_k=1),
                "top5": self.mean_top_k(distances_full, top_k=5),
                "top10": self.mean_top_k(distances_full, top_k=10),
                "top1_percent": self.mean_top_percentile(distances_full, percentile=0.99),
                "top5_percent": self.mean_top_percentile(distances_full, percentile=0.95),
                "top10_percent": self.mean_top_percentile(distances_full, percentile=0.90),
                "softmax_weighted": self.softmax_weighted_mean(distances_full, top_ratio=0.05, tau=0.1)
            }
            
            anomaly_map = distances_full.view(b, 1, *grid_size)
            
            if self.full_anomaly_map:
                anomaly_map = self.anomaly_map_generator(anomaly_map, (cropped_h, cropped_w)) # [B, 1, H, W] - the pixel-level anomaly heatmap for each image, upsampled to the original image size (before cropping).
                
                if crop_h > 0 or crop_w > 0:
                    anomaly_map = F.pad(anomaly_map, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

            results.append({
                "k": k,
                "pred_scores": scores_dict,
                "anomaly_map": anomaly_map
            })
            
        return results if len(results) > 1 else results[0]