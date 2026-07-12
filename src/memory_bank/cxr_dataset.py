import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_valid_transform
from src.memory_bank.utils import load_anatomy_masking

class NormalCXRDataset(Dataset):
    """
    Dataset class for loading normal chest X-ray images for memory bank construction.
    Args:
        - normal_image_dir (str): Directory containing normal chest X-ray images.
        - size (int): Maximum number of images to include in the dataset for memory bank construction
        - data_tail (bool): If True, use the last 'size' images instead of the first 'size' images.
        - anatomy_map (str): If provided, path to the anatomy map dir to filter images based on anatomical regions.
    """
    def __init__(self, normal_image_dir, size=5000, seed = None, anatomy_dir: str = None):
        self.img_dir = normal_image_dir
        self.image_paths = [os.path.join(self.img_dir, fname) for fname in os.listdir(self.img_dir) if fname.endswith('.png')]
        
        # Limit to specified size for memory bank construction
        # if not data_tail:
        #     self.image_paths = self.image_paths[:min(size, len(self.image_paths))]
        # else:
        #     self.image_paths = self.image_paths[-min(size, len(self.image_paths)):]
        
        if seed: 
            self.image_paths = self.image_paths[size * seed: size * (seed + 1)]  # Select a subset of images based on the seed and size
            
        self.anatomy_dir = anatomy_dir
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = Path(img_path).stem  # Get the image ID without extension
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
        
        # Apply valid transformations (e.g., resizing, normalization)
        transform = get_valid_transform()
        image = transform(image=image, bboxes=[], labels=[])['image']
        
        if self.anatomy_dir:
            anatomy_npz_path = os.path.join(self.anatomy_dir, f"{image_id}.npz")
            anatomy_masks = load_anatomy_masking(anatomy_npz_path)
            # Example: Filter based on lung region (assuming 'lung_mask' is a binary mask in the npz file)
            
            filtered_masks = anatomy_masks[[0, 2, 4, 6, 7], :, :] # Corresponding to clavicle, lung, heart, facies diaphragmatica, mediastinum
            tensor_masks = torch.from_numpy(filtered_masks)

            return image, tensor_masks
            
        return image