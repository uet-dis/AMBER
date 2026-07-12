import torch
import cv2
import argparse
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.memory_bank.anomaly_raddino import AnomalyRadDINO
from src.memory_bank.anomaly_raddino_v2 import AnomalyRadDINOv2
from src.memory_bank.utils import generate_online_anatomy_masks, SELECTED_ANATOMIES
import torchxrayvision as xrv
from src.utils import load_raddino, get_valid_transform

def init_anomaly_model_for_inference(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    num_neighbours: int | list[int] = 1
    ):
    """
    Initializes the AnomalyRadDINOv1 model for inference by loading the pre-built memory bank from disk.
    Args:   
    - raddino_checkpoint_path (str): File path to the pre-trained RadDINO model checkpoint (e.g., "raddino_pretrained.pth").
    - memory_bank_path (str): File path to the saved memory bank tensor (e.g., "raddino_memory_bank.pt").
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    raddino = load_raddino(raddino_checkpoint_path)
    
    model = AnomalyRadDINO(raddino_model=raddino, patch_size=16, num_neighbours=num_neighbours)
    
    print(f"Loading Memory Bank from {memory_bank_path}...")
    loaded_bank = torch.load(memory_bank_path, map_location='cpu')  # Load on CPU first to avoid GPU memory issues
    
    # Convert loaded memory bank to float32 for inference, as the model expects float32 tensors for distance calculations
    model.memory_bank = loaded_bank.to(torch.float32) .to(device)  # Move to GPU if available
    
    model = model.to(device)
    
    # Turn on evaluation mode to ensure the model uses the memory bank for anomaly detection during inference
    model.eval() 
    
    return model, device

def init_aamb_model_for_inference(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    num_neighbours: int | list[int] = 1,
    num_anatomies: int = 5,
    full_anomaly_map: bool = True
):
    """
    Initializes the AnomalyRadDINOv2 (AAMB) model for inference by loading the pre-built memory banks from disk.
    Args:
    - raddino_checkpoint_path (str): File path to the pre-trained RadDINO model checkpoint (e.g., "raddino_pretrained.pth").
    - memory_bank_path (str): File path to the saved memory bank tensor (e.g., "aamb_memory_bank.pt").
    - num_neighbours (int or list[int]): Number of nearest neighbors to consider for anomaly scoring. Can be a single integer or a list of integers for multiple settings.
    - num_anatomies (int): Number of anatomical regions considered in the AAMB model (default is 5 for the selected anatomies in the CXR dataset).
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    raddino = load_raddino(raddino_checkpoint_path)
    
    model = AnomalyRadDINOv2(raddino_model=raddino, patch_size=16, num_neighbours=num_neighbours, num_anatomies=num_anatomies, full_anomaly_map=full_anomaly_map)
    
    print(f"Loading AAMB Memory Banks from {memory_bank_path}...")
    loaded_banks = torch.load(memory_bank_path, map_location='cpu')
    
    # Convert loaded memory banks to float32 for inference, as the model expects float32 tensors for distance calculations
    float32_banks = {k: v.to(torch.float32) for k, v in loaded_banks.items()}
    
    model.load_state_dict(float32_banks, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, device

def run_inference(model, device, image_tensor: torch.Tensor):
    """
    Runs inference on a single image tensor using the AnomalyRadDINOv1 model, 
    returning the anomaly map and image score.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension if missing
    
    image_tensor = image_tensor.to(device)
    
    with torch.inference_mode():
        # The model's forward method will compute the anomaly map and image score based on the input image tensor and the loaded memory bank.
        out = model(image_tensor) 
        
    anomaly_map = out['anomaly_map'].squeeze().cpu().numpy() 
    image_score = out["pred_scores"]['top10_percent'].squeeze().cpu().tolist()       
    
    return anomaly_map, image_score

def run_inference_aamb(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    image_path: str,
    num_neighbours: int | list[int] = 1,
    aggregator_strategy: str = 'top10_percent',
):
    """
    Runs inference on a single image using the AnomalyRadDINOv2 (AAMB) model, 
    returning the full anomaly map, anatomy-specific anomaly maps, and the overall image anomaly score.
    """
    model, device = init_aamb_model_for_inference(
        raddino_checkpoint_path=raddino_checkpoint_path,
        memory_bank_path=memory_bank_path,
        num_neighbours=num_neighbours,
        num_anatomies=len(SELECTED_ANATOMIES),
        full_anomaly_map=True
    )
    
    seg_model = init_anatomy_segmentation_model(device)
    
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    transform = get_valid_transform()
    image_raddino = transform(image=image_rgb, bboxes=[], labels=[])['image'].unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    img_xrv = xrv.datasets.normalize(image_gray, 255) # Scale to [-1024, 1024]
    img_xrv = img_xrv[None, ...]                      # Add channel dimension: [1, H, W]
    img_xrv = xrv.datasets.XRayCenterCrop()(img_xrv) # Apply XRayCenterCrop transformation
    img_xrv = torch.from_numpy(img_xrv).float().to(device).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    with torch.inference_mode():
        mask = generate_online_anatomy_masks(img_xrv, seg_model) # Get anatomical masks from the segmentation model
        out = model(image_raddino, mask) # Run the AAMB model with the input image and anatomical masks
        
    full_anomaly_map = out['anomaly_map'].squeeze().cpu().numpy()
    image_score = out['pred_scores'][aggregator_strategy].squeeze().cpu()
    
    return full_anomaly_map, image_score
    

def init_anatomy_segmentation_model(device):
    """
    Initializes a pre-trained anatomy segmentation model (e.g., PSPNet) for use in the AAMB pipeline to generate anatomical masks during inference.
    Args:
        - device: Device to load the segmentation model onto (e.g., 'cuda' or 'cpu').
    """
    
    seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device)
    
    seg_model.eval()
    
    return seg_model

# def get_args_parser(add_help=True):
#     parser = argparse.ArgumentParser(description="Anomaly Detection Inference Script", add_help=add_help)
#     parser.add_argument("--raddino_checkpoint_path", type=str, required=True, help="Path to the pre-trained RadDINO model checkpoint (e.g., 'raddino_pretrained.pth').")
#     parser.add_argument("--memory_bank_path", type=str, required=True, help="Path to the saved memory bank tensor (e.g., 'raddino_memory_bank.pt').")
#     parser.add_argument("--input_image_path", type=str, required=True, help="Path to the input image for anomaly detection (e.g., 'test_image.png').")
#     return parser

# if __name__ == "__main__":
#     args = get_args_parser().parse_args()
    
#     model, device = init_anomaly_model_for_inference(args.raddino_checkpoint_path, args.memory_bank_path)
    
#     image = cv2.imread(args.input_image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
    
#     transform = get_valid_transform()
#     image = transform(image=image, bboxes=[], labels=[])['image']  # Apply the same transformations used during memory bank construction to ensure consistency
    
#     anomaly_map, image_score = run_inference(model, device, image)
    
#     print(f"Anomaly Score: {image_score}")
#     # You can also visualize the anomaly map as needed