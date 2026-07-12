import copy
from torch.utils.data import DataLoader, Subset
import concurrent.futures
from tqdm.auto import tqdm
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.memory_bank.anomaly_detection import init_anomaly_model_for_inference, init_aamb_model_for_inference, init_anatomy_segmentation_model
from src.memory_bank.binary_cxr_dataset import BinaryCXRDataset
from src.memory_bank.utils import generate_online_anatomy_masks, SELECTED_ANATOMIES

def get_model_scores(loader, model, device, desc="Running Inference"):
    """
    Helper function to run inference on the DataLoader and collect predicted scores for multiple neighbor settings.
    Args:
    - loader (DataLoader): DataLoader for the binary classification dataset.
    - model: AnomalyRadDINOv1 model initialized for inference with multiple neighbor settings.
    - device: Device to run inference on (e.g., 'cuda' or 'cpu').
    Return:
    - true_labels: Numpy array of true binary labels for the dataset.
    - predicted_scores_per_k: Dictionary mapping each k value to a numpy array of predicted anomaly scores for the dataset.
    """
    all_labels = []
    k_values = model.num_neighbours
    strategies = ["top1", "top5", "top10", "top1_percent", "top5_percent", "top10_percent", "softmax_weighted"]
    
    all_scores = {k: {s: [] for s in strategies} for k in k_values}
    
    model = model.to(device)
    model.eval()
    
    pos = int(str(device)[-1]) if 'cuda:' in str(device) else 0
    
    for images, labels in tqdm(loader, desc=f"{desc} ({device})", position=pos, leave=True):
        images = images.to(device)
    
        with torch.inference_mode():
            outputs = model(images)
        
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        all_labels.extend(labels.tolist())
        
        for output in outputs:
            k = output["k"]
            for strat_name, score_tensor in output["pred_scores"].items():
                batch_scores = score_tensor.squeeze().cpu().tolist()
                
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                    
                all_scores[k][strat_name].extend(batch_scores)
            
    true_labels = np.array(all_labels)
    
    predicted_scores_per_k_and_strat = {
        k: {s: np.array(scores) for s, scores in strats.items()}
        for k, strats in all_scores.items()
    }
    
    return true_labels, predicted_scores_per_k_and_strat

def get_model_scores_aamb(loader, model, seg_model, device, desc="Running Inference", active_anatomies_idx=None):
    """
    Helper function to run inference on the DataLoader and collect predicted scores for the AAMB model, which includes online anatomy mask generation.
    Args:
    - loader (DataLoader): DataLoader for the binary classification dataset.
    - model: AnomalyRadDINOv2 model initialized for inference with multiple neighbor settings and anatomy segmentation support.
    - seg_model: Pre-trained anatomy segmentation model to generate anatomical masks during inference.
    - device: Device to run inference on (e.g., 'cuda' or 'cpu').
    - active_anatomies_idx: List of indices corresponding to the anatomies to be used for generating anatomical masks. If None, all anatomies will be used.
    """
    all_labels = []
    k_values = model.num_neighbours
    strategies = ["top1", "top5", "top10", "top1_percent", "top5_percent", "top10_percent", "softmax_weighted"]
    all_scores = {k: {s: [] for s in strategies} for k in k_values}
    
    model = model.to(device)
    model.eval()
    
    if seg_model is not None:
        seg_model = seg_model.to(device)
    else:
        raise ValueError("Segmentation model is required for AAMB inference to generate anatomical masks.")
        
    pos = int(str(device)[-1]) if 'cuda:' in str(device) else 0
    
    for images, images_xrv, labels in tqdm(loader, desc=f"{desc} ({device})", position=pos, leave=True):
        
        images = images.to(device)
        images_xrv = images_xrv.to(device)
    
        with torch.inference_mode():
            masks = generate_online_anatomy_masks(images_xrv, seg_model)
            outputs = model(images, anatomy_masks=masks, active_anatomies_idx=active_anatomies_idx)

        if not isinstance(outputs, list):
            outputs = [outputs]
            
        all_labels.extend(labels.tolist())
        
        for output in outputs:
            k = output["k"]
            for strat_name, score_tensor in output["pred_scores"].items():
                batch_scores = score_tensor.squeeze().cpu().tolist()
                batch_scores = [batch_scores] if isinstance(batch_scores, float) else batch_scores
                all_scores[k][strat_name].extend(batch_scores)
            
    true_labels = np.array(all_labels)
    predicted_scores = {
        k: {s: np.array(scores) for s, scores in strats.items()} 
        for k, strats in all_scores.items()
    }
    return true_labels, predicted_scores

def get_aamb_anomaly_maps(loader, model, seg_model, device, desc="Generating Anomaly Maps"):
    """
    Helper function to run inference on the DataLoader and generate anomaly maps for the AAMB model.
    Args:
    - loader (DataLoader): DataLoader for the binary classification dataset.
    - model: AnomalyRadDINOv2 model initialized for inference with multiple neighbor settings and anatomy segmentation support.
    - seg_model: Pre-trained anatomy segmentation model to generate anatomical masks during inference.
    - device: Device to run inference on (e.g., 'cuda' or 'cpu').
    Return:
    - anomaly_maps: List of generated anomaly maps for each input image in the dataset.
    """
    all_anomaly_maps = []
    all_img_ids = []
    all_anomaly_scores = []
    
    model = model.to(device)
    model.eval()
    
    if seg_model is not None:
        seg_model = seg_model.to(device)
    else:
        raise ValueError("Segmentation model is required for AAMB inference to generate anatomical masks.")
        
    pos = int(str(device)[-1]) if 'cuda:' in str(device) else 0
    
    for images, images_xrv, _, img_ids in tqdm(loader, desc=f"{desc} ({device})", position=pos, leave=True):
        
        images = images.to(device)
        images_xrv = images_xrv.to(device)
    
        with torch.inference_mode():
            masks = generate_online_anatomy_masks(images_xrv, seg_model)
            output = model(images, anatomy_masks=masks)

        if isinstance(output, list):
            raise ValueError("Model outputs should be a single dictionary containing the 'anomaly_map' key as setting only one K value (K = 1)")
            
        all_anomaly_maps.extend(output["anomaly_map"].cpu().numpy())
        anomaly_scores = output["pred_scores"]["top10_percent"].squeeze().cpu().tolist()
        anomaly_scores = [anomaly_scores] if isinstance(anomaly_scores, float) else anomaly_scores
        all_anomaly_scores.extend(anomaly_scores)
        all_img_ids.extend(img_ids)
            
    return all_img_ids, all_anomaly_maps, all_anomaly_scores

def binary_classification_inference(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    normal_paths: list,
    abnormal_paths: list,
    batch_size: int = 16,
    num_neighbours: list[int] = [1, 2, 4, 8, 10],
):
    """
    Main function to run binary classification inference using the AnomalyRadDINO model with support for multiple neighbor settings and multi-GPU parallelism.
    Args:
    - raddino_checkpoint_path: Path to the RadDINO checkpoint for model initialization.
    - memory_bank_path: Path to the memory bank file for model initialization.
    - normal_paths: List of file paths for normal (negative) samples.
    - abnormal_paths: List of file paths for abnormal (positive) samples.
    - batch_size: Batch size for DataLoader during inference.
    - num_neighbours: List of k values for the number of neighbors to compute anomaly scores for. The model should be initialized to support these k values.
    Return:
    - true_labels: Numpy array of true binary labels for the dataset.
    - predicted_scores_per_k: Dictionary mapping each k value to a numpy array of predicted anomaly scores for the dataset.
    """
    
    model_0, device_0 = init_anomaly_model_for_inference(
        raddino_checkpoint_path,
        memory_bank_path,
        num_neighbours=num_neighbours,
    )

    dataset = BinaryCXRDataset(normal_paths, abnormal_paths)

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Running multi-threading inference manually...")
        device_1 = torch.device('cuda:1')

        print("Cloning model to cuda:1...")
        model_1 = copy.deepcopy(model_0).to(device_1)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        mid = dataset_size // 2

        subset_0 = Subset(dataset, indices[:mid])
        subset_1 = Subset(dataset, indices[mid:])
        
        loader_0 = DataLoader(subset_0, batch_size=batch_size, shuffle=True, num_workers=2)
        loader_1 = DataLoader(subset_1, batch_size=batch_size, shuffle=True, num_workers=2)

        print("Starting parallel evaluation...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(get_model_scores, loader_0, model_0, device_0, "Inference Part 1")
            future_1 = executor.submit(get_model_scores, loader_1, model_1, device_1, "Inference Part 2")
    
            labels_0, scores_0 = future_0.result()
            labels_1, scores_1 = future_1.result()

        print("\nCombining results...")
        true_labels = np.concatenate([labels_0, labels_1])
        
        strategies = ["top1", "top5", "top10", "top1_percent", "top5_percent", "top10_percent", "softmax_weighted"]
        predicted_scores_per_k = {}

        for k in num_neighbours:
            predicted_scores_per_k[k] = {}
            for s in strategies:
                predicted_scores_per_k[k][s] = np.concatenate([scores_0[k][s], scores_1[k][s]])
    else:
        print("Detected 1 GPU. Running standard inference...")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        true_labels, predicted_scores_per_k = get_model_scores(loader, model_0, device_0, "Inference")

    return true_labels, predicted_scores_per_k

def binary_classification_inference_aamb(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    normal_paths: list,
    abnormal_paths: list,
    batch_size: int = 16,
    num_neighbours: list[int] = [1, 2, 4, 8, 10, 15, 20],
    active_anatomies: list[str] = None
):
    """
    Run binary classification inference using the AnomalyRadDINOv2 (AAMB) model with support for multiple neighbor settings, online anatomy mask generation, and multi-GPU parallelism.
    Args:
    - raddino_checkpoint_path: Path to the RadDINO checkpoint for model initialization.
    - memory_bank_path: Path to the memory bank file for model initialization.
    - normal_paths: List of file paths for normal (negative) samples.
    - abnormal_paths: List of file paths for abnormal (positive) samples.
    - batch_size: Batch size for DataLoader during inference.
    - num_neighbours: List of k values for the number of neighbors to compute anomaly scores
    - active_anatomies: List of anatomy names to be used for generating anatomical masks. If None, all anatomies will be used.
    """
    
    dataset = BinaryCXRDataset(normal_paths, abnormal_paths, use_aamb=True)
    
    model_0, device_0 = init_aamb_model_for_inference(
        raddino_checkpoint_path, 
        memory_bank_path, 
        num_neighbours, 
        num_anatomies=len(SELECTED_ANATOMIES)
    )
    
    seg_model_0 = init_anatomy_segmentation_model(device_0)
    
    # Filtering active_anatomies_idx based on provided active_anatomies list
    active_anatomies_idx = None
    if active_anatomies is not None:
        active_anatomies_idx = []
        for anatomy in active_anatomies:
            if anatomy in SELECTED_ANATOMIES:
                active_anatomies_idx.append(SELECTED_ANATOMIES.index(anatomy))
            else:
                raise ValueError(f"Anatomy '{anatomy}' is not in the list of selected anatomies: {SELECTED_ANATOMIES}")
    
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Running multi-threading inference manually...")
        
        device_1 = torch.device('cuda:1')
        model_1 = copy.deepcopy(model_0).to(device_1)
        seg_model_1 = copy.deepcopy(seg_model_0).to(device_1)

        dataset_size = len(dataset)
        mid = dataset_size // 2
        subset_0 = Subset(dataset, list(range(dataset_size))[:mid])
        subset_1 = Subset(dataset, list(range(dataset_size))[mid:])
        
        loader_0 = DataLoader(subset_0, batch_size=batch_size, shuffle=False, num_workers=2)
        loader_1 = DataLoader(subset_1, batch_size=batch_size, shuffle=False, num_workers=2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(get_model_scores_aamb, loader_0, model_0, seg_model_0, device_0, "Inference Part 1", active_anatomies_idx)
            future_1 = executor.submit(get_model_scores_aamb, loader_1, model_1, seg_model_1, device_1, "Inference Part 2", active_anatomies_idx)
    
            labels_0, scores_0 = future_0.result()
            labels_1, scores_1 = future_1.result()

        print("\nCombining results...")
        true_labels = np.concatenate([labels_0, labels_1])
        
        strategies = ["top1", "top5", "top10", "top1_percent", "top5_percent", "top10_percent", "softmax_weighted"]
        predicted_scores_per_k = {}
        for k in num_neighbours:
            predicted_scores_per_k[k] = {}
            for s in strategies:
                predicted_scores_per_k[k][s] = np.concatenate([scores_0[k][s], scores_1[k][s]])
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        true_labels, predicted_scores_per_k = get_model_scores_aamb(loader, model_0, seg_model_0, device_0, "Inference")

    return true_labels, predicted_scores_per_k

def anomaly_map_generator(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    normal_paths: list,
    abnormal_paths: list,
    batch_size: int = 16,
    num_neighbours: list[int] = [1],
    full_anomaly_map=False
):
    """
    Run inference using the AnomalyRadDINOv2 (AAMB) model to generate patch-level anomaly maps for each input image in the dataset.
    Args:
    - raddino_checkpoint_path: Path to the RadDINO checkpoint for model initialization.
    - memory_bank_path: Path to the memory bank file for model initialization.
    - normal_paths: List of file paths for normal (negative) samples.
    - abnormal_paths: List of file paths for abnormal (positive) samples.
    - batch_size: Batch size for DataLoader during inference.
    - num_neighbours: List of k values for the number of neighbors to compute anomaly scores. For anomaly map generation, typically only K=1 is used.
    - full_anomaly_map: Boolean flag indicating whether to return full anomaly maps or patch-level anomaly maps. If True, the model will return full anomaly maps; if False, it will return patch-level anomaly maps.
    """
    
    dataset = BinaryCXRDataset(normal_paths, abnormal_paths, use_aamb=True, return_img_id=True)
    model_0, device_0 = init_aamb_model_for_inference(
        raddino_checkpoint_path, 
        memory_bank_path, 
        num_neighbours, 
        num_anatomies=len(SELECTED_ANATOMIES),
        full_anomaly_map=full_anomaly_map
    )
    
    seg_model_0 = init_anatomy_segmentation_model(device_0)
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Running multi-threading inference manually...")
        
        device_1 = torch.device('cuda:1')
        model_1 = copy.deepcopy(model_0).to(device_1)
        seg_model_1 = copy.deepcopy(seg_model_0).to(device_1)
        
        dataset_size = len(dataset)
        mid = dataset_size // 2
        subset_0 = Subset(dataset, list(range(dataset_size))[:mid])
        subset_1 = Subset(dataset, list(range(dataset_size))[mid:])
        
        loader_0 = DataLoader(subset_0, batch_size=batch_size, shuffle=False, num_workers=2)
        loader_1 = DataLoader(subset_1, batch_size=batch_size, shuffle=False, num_workers=2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(get_aamb_anomaly_maps, loader_0, model_0, seg_model_0, device_0, "Inference Part 1")
            future_1 = executor.submit(get_aamb_anomaly_maps, loader_1, model_1, seg_model_1, device_1, "Inference Part 2")
    
            img_ids_0, anomaly_maps_0, anomaly_scores_0 = future_0.result()
            img_ids_1, anomaly_maps_1, anomaly_scores_1 = future_1.result()
            
        print("\nCombining results...")
        all_img_ids = np.concatenate([img_ids_0, img_ids_1])
        all_anomaly_maps = np.concatenate([anomaly_maps_0, anomaly_maps_1])
        all_anomaly_scores = np.concatenate([np.array(anomaly_scores_0), np.array(anomaly_scores_1)])
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_img_ids, all_anomaly_maps, all_anomaly_scores = get_aamb_anomaly_maps(loader, model_0, seg_model_0, device_0, "Inference")  
        all_anomaly_scores = np.array(all_anomaly_scores) 
        
    return all_img_ids, all_anomaly_maps, all_anomaly_scores