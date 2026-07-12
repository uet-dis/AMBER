import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.memory_bank.anomaly_raddino import AnomalyRadDINO
from src.memory_bank.anomaly_raddino_v2 import AnomalyRadDINOv2
from src.memory_bank.cxr_dataset import NormalCXRDataset
from src.memory_bank.utils import SELECTED_ANATOMIES
from src.utils import load_raddino

def build_and_save_memory_bank(
    raddino_checkpoint_path: str,
    train_normal_loader: DataLoader, 
    save_path: str = "raddino_memory_bank.pt",
    subsampling_ratio: float = 0.01,
    normal_num: int = 1000
):
    """
    Builds a single memory bank (AnomalyRadDINOv1)
    
    Args:
    - raddino_checkpoint_path (str): File path to the interpolated pre-trained RadDINO model checkpoint.
    - train_normal_loader (DataLoader): DataLoader for the normal training dataset, yielding batches of images.
    - save_path (str): File path to save the resulting memory bank tensor.
    - subsampling_ratio (float): Ratio for coreset subsampling to reduce memory bank size (e.g., 0.01 for 1% of the original size). 
    Adjust based on available RAM and dataset size.
    """

    # Load pre-trained RadDINO model (to be implemented as needed)
    raddino = load_raddino(raddino_checkpoint_path)
    
    # Initialize AnomalyRadDINO model with the pre-trained RadDINO feature encoder
    model = AnomalyRadDINO(
        raddino_model=raddino,
        patch_size=16,
        coreset_subsampling=True,
        sampling_ratio=subsampling_ratio
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for memory bank construction.")
        model.feature_encoder = nn.DataParallel(model.feature_encoder)
        
    model = model.cuda()
    model.train()
    
    print("[V1] Memory Bank Construction Started...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(train_normal_loader)):
            
            images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            
            images = images.cuda()
            
            _ = model(images)
            
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
            if len(model.embedding_store) > 0:
                model.embedding_store[-1] = model.embedding_store[-1].cpu()
            
    print(f"Extraction Completed. Total patch batches collected: {len(model.embedding_store)}.")
    
    torch.cuda.empty_cache()
    
    print("Starting Coreset Subsampling")
    model.fit(normal_num=normal_num)
    
    memory_bank_cpu = model.memory_bank.cpu()
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(memory_bank_cpu, save_path)
    
    print(f"Save memory bank at: {save_path}")
    print(f"The coreset shape: {memory_bank_cpu.shape}")


def build_and_save_anatomical_aware_bank(
    raddino_checkpoint_path: str,
    train_normal_loader: DataLoader, 
    save_path: str = "raddino_aamb.pt",
    subsampling_ratio: float = 0.01,
):
    """
    Builds anatomical-aware memory bank (AnomalyRadDINOv2)
    
    Args:
    - raddino_checkpoint_path (str): File path to the interpolated pre-trained RadDINO model checkpoint.
    - train_normal_loader (DataLoader): DataLoader for the normal training dataset, yielding batches of images.
    - save_path (str): File path to save the resulting memory bank tensor.
    - subsampling_ratio (float): Ratio for coreset subsampling to reduce memory bank size (e.g., 0.01 for 1% of the original size).
        Adjust based on available RAM and dataset size.
    - normal_num (int): Number of normal samples to use for memory bank construction (if dataset is large).
    """
    raddino = load_raddino(raddino_checkpoint_path)
    
    model = AnomalyRadDINOv2(
        raddino_model=raddino,
        patch_size=16,
        coreset_subsampling=True,
        sampling_ratio=subsampling_ratio,
        num_anatomies=len(SELECTED_ANATOMIES)
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for AAMB construction.")
        model.feature_encoder = nn.DataParallel(model.feature_encoder)
    
    model = model.cuda()
    model.train()
    
    print("[V2] Anatomical-Aware Memory Bank Construction Started...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(train_normal_loader)):
            
            images, masks = batch_data
            images = images.cuda()
            masks = masks.cuda()
            
            _ = model(images, masks)
            
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
            for c in range(model.num_anatomies):
                if len(model.embedding_stores[c]) > 0:
                    model.embedding_stores[c][-1] = model.embedding_stores[c][-1].cpu()
    
    print(f"Multi-organ Extraction Completed.")
    torch.cuda.empty_cache()
    
    print("Starting independent Coreset Subsampling for Sub-Banks...")
    model.fit()
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    state_to_save = {k: v.cpu() for k, v in model.state_dict().items() if 'memory_bank' in k}
    
    torch.save(state_to_save, save_path)
    print(f"Saved AAMB, including full of sub-banks at: {save_path}")
    
    for c in range(model.num_anatomies):
        k = f'memory_bank_{c}'
        if k in state_to_save:
            print(f"Organ {SELECTED_ANATOMIES[c]} coreset shape: {state_to_save[k].shape}")
    
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Build Memory Bank for Anomaly Detection", add_help=add_help)
    parser.add_argument("--raddino_checkpoint_path", type=str, required=True, help="Path to the pre-trained RadDINO checkpoint.")
    parser.add_argument("--train_normal_data_dir", type=str, required=True, help="Directory containing normal training images.")
    parser.add_argument("--save_path", type=str, default="raddino_memory_bank.pt", help="File path to save the memory bank tensor.")
    parser.add_argument("--subsampling_ratio", type=float, default=0.05, help="Ratio for coreset subsampling (e.g., 0.01 for 1%).")
    parser.add_argument("--normal_num", type=int, default=1000, help="Number of normal samples to use for memory bank construction (if dataset is large).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    # parser.add_argument("--data_tail", action='store_true', help="Whether to use the tail of the dataset for memory bank construction (useful if data is ordered by time).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for selecting a subset of images from the dataset (useful for large datasets).")
    
    # New arguments for anatomical-aware memory bank construction
    parser.add_argument("--use_aamb", action="store_true", help="Launch AnomalyRadDINOv2 with Sub-Banks.")
    parser.add_argument("--anatomy_dir", type=str, default=None, help="Directory containing '.npz' mask files (Required if --use_aamb).")
    
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    if args.use_aamb and not args.anatomy_dir:
        raise ValueError("--anatomy_dir is heavily required when --use_aamb flag is on.")
    
    # Prepare DataLoader for normal training dataset
    train_normal_dataset = NormalCXRDataset(
        args.train_normal_data_dir, 
        size=args.normal_num, 
        # data_tail=args.data_tail,
        seed=args.seed,
        anatomy_dir=args.anatomy_dir if args.use_aamb else None
    )
    
    train_normal_loader = DataLoader(train_normal_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    if not args.use_aamb:
        build_and_save_memory_bank(
            raddino_checkpoint_path=args.raddino_checkpoint_path,
            train_normal_loader=train_normal_loader,
            save_path=args.save_path,
            subsampling_ratio=args.subsampling_ratio,
            normal_num=args.normal_num
        )
    else:
        build_and_save_anatomical_aware_bank(
            raddino_checkpoint_path=args.raddino_checkpoint_path,
            train_normal_loader=train_normal_loader,
            save_path=args.save_path,
            subsampling_ratio=args.subsampling_ratio,
        )