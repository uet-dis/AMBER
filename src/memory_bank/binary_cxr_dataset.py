from torch.utils.data import Dataset
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_valid_transform
import torchvision
import torchxrayvision as xrv
import torch

class BinaryCXRDataset(Dataset):
    """
    Dataset class for loading both normal and abnormal chest X-ray images for anomaly detection.
    Args:
    - normal_paths: List of file paths to normal chest X-ray images.
    - abnormal_paths: List of file paths to abnormal chest X-ray images.
    - use_aamb: Boolean flag indicating whether to use AnomalyRadDINOv2 (AAMB), if true the dataset
        will return 2 version of image tensors: one for RadDINO extracts features and one for PSPNet extracts anatomy masks.
        If false, the dataset will return only the first version.
    """
    def __init__(self, normal_paths, abnormal_paths, use_aamb=False, return_img_id=False):
        self.image_paths = normal_paths + abnormal_paths
        self.labels = [0] * len(normal_paths) + [1] * len(abnormal_paths)
        self.use_aamb = use_aamb
        
        if self.use_aamb:
            self.xrv_transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
            ])
            
        self.return_img_id = return_img_id
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = Path(img_path).stem
        label = self.labels[idx]
        
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
        
        # Apply valid transformations (e.g., resizing, normalization)
        transform = get_valid_transform()
        image_raddino = transform(image=image_rgb, bboxes=[], labels=[])['image']
        
        if self.use_aamb:
            img_xrv = xrv.datasets.normalize(image_gray, 255) # Scale về [-1024, 1024]
            img_xrv = img_xrv[None, ...]                      # Thêm chiều Channel: [1, H, W]
            img_xrv = self.xrv_transform(img_xrv)             # Áp dụng XRayCenterCrop()
            img_xrv = torch.from_numpy(img_xrv).float()
            
            if self.return_img_id:  
                return image_raddino, img_xrv, label, img_id
            
            return image_raddino, img_xrv, label
        
        if self.return_img_id:
            return image_raddino, label, img_id
        
        return image_raddino, label
        
    