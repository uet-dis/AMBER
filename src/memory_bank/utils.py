import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_valid_transform, get_bounding_boxes

ORIGINAL_ANATOMIES = ['Left Clavicle',
 'Right Clavicle',
 'Left Scapula',
 'Right Scapula',
 'Left Lung',
 'Right Lung',
 'Left Hilus Pulmonis',
 'Right Hilus Pulmonis',
 'Heart',
 'Aorta',
 'Facies Diaphragmatica',
 'Mediastinum',
 'Weasand',
 'Spine']

MERGED_ANATOMIES = [
    'Clavicle',            # Merging index 0, 1
    'Scapula',             # Merging index 2, 3
    'Lung',                # Merging index 4, 5
    'Hilus Pulmonis',      # Merging index 6, 7
    'Heart',               # Index 8
    'Aorta',               # Index 9
    'Facies Diaphragmatica', # Index 10
    'Mediastinum',         # Index 11
    'Weasand',             # Index 12
    'Spine'  
]

SELECTED_ANATOMIES = ['Clavicle', 'Lung', 'Heart', 'Facies Diaphragmatica', 'Mediastinum'] # Corresponding to merged indices 0, 2, 4, 6, 7

def _normalize_color_for_matplotlib(c):
    """Normalize color to [0, 1] range for matplotlib."""
    
    c = np.array(c, dtype=np.float32)
    if c.max() > 1.0:
        c = c / 255.0
    return tuple(c.tolist())

def _draw_boxes_xyxy(ax, bboxes, labels, colors, linewidth=2):
    """Draw bounding boxes on a matplotlib axis."""
    
    for (x1, y1, x2, y2), label, color in zip(bboxes, labels, colors):
        
        color = _normalize_color_for_matplotlib(color)

        rect = plt.Rectangle(
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            fill=False,
            edgecolor=color,
            linewidth=linewidth
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 4),
            str(label),
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.75, edgecolor="none", pad=1.5)
        )
        
def _read_image(input_image_path):
    """Reads an image from disk, converts it to RGB, 
    and applies the same transformations used during memory bank construction.
    """
    
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
    
    transform = get_valid_transform()
    image = transform(image=image, bboxes=[], labels=[])['image']  # Apply the same transformations used during memory bank construction to ensure consistency

    return image

def load_anatomy_masking(npz_path: str):
    """Loads the anatomy masks from the given npz file and merges them into 10 regions based on the predefined mapping.
    Args:
    - npz_path (str): Path to the npz file containing the original 14 anatomy masks.
    """

    data = np.load(npz_path)
    
    raw_mask = data['mask']

    merged_mask = np.zeros((10, raw_mask.shape[1], raw_mask.shape[2]), dtype=np.bool_)

    merged_mask[0] = raw_mask[0] | raw_mask[1]  # Left + Right Clavicle
    merged_mask[1] = raw_mask[2] | raw_mask[3]  # Left + Right Scapula
    merged_mask[2] = raw_mask[4] | raw_mask[5]  # Left + Right Lung
    merged_mask[3] = raw_mask[6] | raw_mask[7]  # Left + Right Hilus
    merged_mask[4] = raw_mask[8]                # Heart
    merged_mask[5] = raw_mask[9]                # Aorta
    merged_mask[6] = raw_mask[10]               # Facies Diaphragmatica
    merged_mask[7] = raw_mask[11]               # Mediastinum
    merged_mask[8] = raw_mask[12]               # Weasand
    merged_mask[9] = raw_mask[13]               # Spine
    
    return merged_mask

def generate_online_anatomy_masks(images_tensor: torch.Tensor, seg_model: torch.nn.Module):
    """ 
    Generates anatomical masks for a batch of input images using the provided segmentation model.
    Args:
        - images_tensor: A batch of input images as a tensor of shape (B, C, H, W).
        - seg_model: A pre-trained segmentation model (e.g., PSPNet) that takes in the input images and outputs anatomical segmentation masks.
    """
    with torch.inference_mode():
        
        imgs_512 = F.interpolate(images_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        
        pred_512 = seg_model(imgs_512)  # (B, num_classes, 512, 512)
        
        pred_1024 = F.interpolate(pred_512, size=(images_tensor.shape[-2], images_tensor.shape[-1]), mode='bilinear', align_corners=False)  # (B, num_classes, 1024, 1024)
        
        pred_soft = 1 / (1 + torch.exp(-pred_1024))
        
        pred_bin = pred_soft > 0.5  # (B, num_classes, H, W), binary masks for each anatomy
        
        b, _, h, w = pred_bin.shape
        
        pred_masks = torch.zeros((b, len(MERGED_ANATOMIES), h, w), dtype=torch.bool, device=images_tensor.device)  # (B, 5, H, W) for the 5 selected anatom
        
        pred_masks[:, 0] = pred_bin[:, 0] | pred_bin[:, 1]  # Clavicle
        pred_masks[:, 1] = pred_bin[:, 2] | pred_bin[:, 3]  # Scapula
        pred_masks[:, 2] = pred_bin[:, 4] | pred_bin[:, 5]  # Lung
        pred_masks[:, 3] = pred_bin[:, 6] | pred_bin[:, 7]  # Hilus Pulmonis
        pred_masks[:, 4] = pred_bin[:, 8]                   # Heart
        pred_masks[:, 5] = pred_bin[:, 9]                   # Aorta
        pred_masks[:, 6] = pred_bin[:, 10]                  # Facies Diaphragmatica
        pred_masks[:, 7] = pred_bin[:, 11]                  # Mediastinum
        pred_masks[:, 8] = pred_bin[:, 12]                  # Weasand
        pred_masks[:, 9] = pred_bin[:, 13]                  # Spine

    
        
    return pred_masks[:, [0, 2, 4, 6, 7], :, :]  # Return only the selected anatomies: Clavicle, Lung, Heart, Facies Diaphragmatica, Mediastinum

def visualize_anomaly_map_aamb(
    raddino_checkpoint_path: str,
    memory_bank_path: str,
    image_path: str,
    annotations_path: str,
    num_neighbours: int = 1,
    aggregator_strategy: str = 'top10_percent',
    alpha: float = 0.45,
    clip_percentiles: tuple = (1, 99.9),
    cmap=cv2.COLORMAP_TURBO,
    gamma: float = 1.0,
    threshold_method: str = "percentile",  # "percentile" | "otsu" | "topk"
    threshold_param: float = 99.0,         
    min_area_frac: float = 0.0005,         
    mean_intensity_th: float = 0.0,        
    morph_kernel_size: int = 5,
    title: str = None,
    show_stats: bool = True,
):
    """
    Visualizes the anomaly map generated by the AAMB model for a given input image, along with the original image, ground-truth bounding boxes, and an overlay of the anomaly map on the original image.
    Args:
    - raddino_checkpoint_path: Path to the checkpoint file for the RadDINO model
    - memory_bank_path: Path to the pre-constructed memory bank file
    - image_path: Path to the input chest X-ray image for inference
    - annotations_path: Path to the annotation file containing ground-truth bounding boxes and labels for
        the input image
    - num_neighbours: Number of nearest neighbors to retrieve from the memory bank for anomaly scoring
    - aggregator_strategy: Strategy to aggregate the anomaly scores from the retrieved neighbors (e.g.,
        'mean', 'top10_percent', etc.)
    - alpha: Transparency factor for the anomaly heatmap overlay (0.0 to 1
    - clip_percentiles: Tuple of (lower_percentile, upper_percentile) to clip the anomaly map values for better visualization
    - cmap: Colormap to use for visualizing the anomaly map (default is cv2
        COLORMAP_TURBO)
    - gamma: Gamma correction factor to apply to the anomaly map for better visualization
    - threshold_method: Method to determine the threshold for creating a binary mask from the anomaly map
    - threshold_param: Parameter for the thresholding method (e.g., percentile value, top
    - min_area_frac: Minimum area fraction for connected components to be kept in the binary mask
    - mean_intensity_th: Minimum mean intensity within a connected component for it to be kept
    - morph_kernel_size: Kernel size for morphological operations to clean up the binary mask
    - title: Optional title for the visualization plot
    - show_stats: Whether to show statistics (mean, max, % above threshold) for
    """
    
    from src.memory_bank.anomaly_detection import run_inference_aamb  # Import the inference function for AAMB
    
    # 1. Chạy Inference (Hàm bạn vừa hoàn thiện phía trên)
    amap, image_score = run_inference_aamb(
        raddino_checkpoint_path=raddino_checkpoint_path,
        memory_bank_path=memory_bank_path,
        image_path=image_path,
        num_neighbours=num_neighbours,
        aggregator_strategy=aggregator_strategy
    )
    
    # Đọc lại ảnh RGB gốc ra [0...1] format chuẩn để visualize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]
    
    # Đảm bảo amap có kích thước trùng gốc
    if amap.shape != (h_img, w_img):
        amap = cv2.resize(amap, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        
    # 2. Xử lý Anomaly Map về chuẩn [0, 1]
    if clip_percentiles is not None:
        low, high = np.percentile(amap, clip_percentiles)
        if high > low:
            pass # Clip handling tuỳ chỉnh 
            
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    if gamma is not None:
        amap = np.power(amap, gamma)

    # 3. Tính toán Threshold Mask
    if threshold_method == "otsu":
        try:
            from skimage.filters import threshold_otsu
            th = threshold_otsu((amap * 255).astype(np.uint8)) / 255.0
        except Exception:
            th = np.percentile(amap, threshold_param if threshold_param else 99.0)
    elif threshold_method == "topk":
        frac = float(threshold_param)
        k = max(1, int(frac * amap.size))
        kth = np.partition(amap.ravel(), -k)[-k]
        th = float(kth)
    else:
        th = np.percentile(amap, float(threshold_param))
        
    mask = (amap >= th).astype(np.uint8) * 255

    # 4. Filter nhe nhiễu (Denoise) bằng Morphology và Area Filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_area_px = max(1, int(min_area_frac * h_img * w_img))
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    kept_contours = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_val = (amap * (cnt_mask > 0)).sum() / max(1, (cnt_mask > 0).sum())
        if mean_val < mean_intensity_th:
            continue
        filtered_mask = np.maximum(filtered_mask, cnt_mask)
        kept_contours.append(cnt)

    # 5. Overlays Heatmap & Vẽ Ground Truth Bounding Box
    amap_u8 = (amap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(amap_u8, cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip((1 - alpha) * img + alpha * heatmap_rgb, 0, 1)

    overlay_disp = (overlay * 255).astype(np.uint8).copy()
    if kept_contours:
        cv2.drawContours(overlay_disp, kept_contours, -1, (255, 0, 0), 2)  
    overlay_disp = cv2.cvtColor(overlay_disp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    bboxes, labels, colors = get_bounding_boxes(image_path=image_path, annotations_path=annotations_path)
    box_stats = []
    for bb in bboxes:
        x1, y1, x2, y2 = map(int, bb)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        if x2 <= x1 or y2 <= y1:
            box_stats.append((0.0, 0.0, 0.0))
            continue
        roi = amap[y1:y2, x1:x2]
        mean_in = float(roi.mean() if roi.size else 0.0)
        max_in = float(roi.max() if roi.size else 0.0)
        pct_above = float((roi >= th).sum() / max(1, roi.size))
        box_stats.append((mean_in, max_in, pct_above))

    # 6. Hiển thị Lên Matplotlib Grid
    from src.memory_bank.utils import _draw_boxes_xyxy  # Tận dụng code hỗ trợ vẽ Bbox
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img)
    _draw_boxes_xyxy(axes[0], bboxes, labels, colors)
    axes[0].set_title("CXR + Ground-truth BBoxes")
    axes[0].axis("off")

    axes[1].imshow(amap, cmap="turbo")
    axes[1].axhline(0, color='none')
    axes[1].set_title(f"AAMB Map (th={th:.4f})")
    axes[1].axis("off")

    axes[2].imshow(overlay_disp)
    _draw_boxes_xyxy(axes[2], bboxes, labels, colors)
    if show_stats:
        for i, bb in enumerate(bboxes):
            x1, y1, x2, y2 = bb
            mean_in, max_in, pct_above = box_stats[i]
            txt = f"m={mean_in:.3f}\nM={max_in:.3f}\n%>{int(pct_above*1000)/10:.1f}"
            axes[2].text(x1, max(0, y1 - 6 - 20 * (i % 3)), txt, color="yellow", fontsize=9,
                         bbox=dict(facecolor="black", alpha=0.5, pad=2))
    axes[2].set_title(f"Overlay + Features (Score: {float(image_score):.4f})")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return {
        "fig": fig,
        "amap": amap,
        "threshold": th,
        "filtered_mask": filtered_mask,
        "box_stats": box_stats,
        "image_score": image_score,
    }