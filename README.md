# AMBER: Anatomy-aware Memory Banks for Efficient Radiographic Anomaly Detection

<!-- <div align="center">

**Few-shot Chest X-ray Anomaly Detection with Medical Foundation Models**

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2406.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.13+](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Paper**](./main.md) • [**Datasets**](#datasets) • [**Benchmarks**](#results) • [**Quick Start**](#quick-start) -->
<!-- 
</div> --> 



## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Datasets](#datasets)
6. [Memory Bank Construction](#memory-bank-construction)
7. [Inference](#inference)
8. [Evaluation & Analysis](#evaluation--analysis)
9. [Notebooks Guide](#notebooks-guide)
<!-- 10. [Results](#results)
11. [Computational Complexity](#computational-complexity)
12. [Citation](#citation) -->

---

## 🔬 Overview

**AMBER** (**A**natomy-aware **M**emory **B**anks for **E**fficient **R**adiographic anomaly detection) is a few-shot unsupervised anomaly detection framework for Chest X-ray (CXR) imaging. 

### The Problem

Conventional memory-bank-based anomaly detection methods organize normal reference features in a **global memory bank**, allowing query patches to retrieve neighbors from any anatomical region. This causes *cross-anatomical semantic mismatch*: 
- A patch from the **lung** might be compared against normal features from the **heart** or **mediastinum**
- This introduces noise and reduces anomaly detection reliability, especially in few-shot settings

### The Solution: AMBER

Instead of a single global bank, AMBER introduces **Anatomy-Aware Memory Banking (AAMB)**:

1. **Anatomical Decomposition**: Normal reference features are organized into **5 organ-specific sub-banks**:
   - Clavicle, Lung, Heart, Diaphragm, Mediastinum

2. **Anatomy-Restricted Retrieval**: Each query patch is matched only against the sub-bank from its anatomical region

3. **Independent Coreset Compression**: Each sub-bank is compressed using K-Center Greedy, reducing memory footprint by ~30% while preserving discriminative normal variations

4. **Foundation Model Backbone**: Uses frozen **RadDINO** (medical self-supervised ViT-B/16) for strong CXR feature extraction

---

## 🎯 Key Contributions

1. **AMBER Framework**: An anatomy-conditioned memory-retrieval architecture for few-shot CXR anomaly detection using frozen foundation model features

2. **Anatomy-Aware Memory Bank (AAMB) Mechanism**: Organizes normal patches into organ-specific sub-banks with restricted nearest-neighbor retrieval to anatomically relevant references

3. **Region-wise Coreset Compression**: Independent K-Center Greedy compression per anatomical region, improving scalability while maintaining discriminative normal variation

4. **Comprehensive Benchmarking**: Systematic evaluation on **VinDr-CXR** and **RSNA** under **MedIAnomaly** and **BMAD** protocols, with:
   - Few-shot performance analysis (N ∈ {1, 5, 10, 20, 50+})
   - Controlled ablations of AAMB
   - Computational cost analysis
   - Failure case analysis

---

## 🔧 Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.13+
- CUDA 11.8+ (GPU recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/uet-dis/AMBER.git
git submodule update --init --recursive
```

### Step 2: Install Dependencies

We use `uv` for fast dependency resolution. If not installed:
```bash
pip install uv
```

Then install all packages in order:

```bash
# Install main dependencies
uv sync

# Install Detectron2 (for Detrex backbone support)
cd detrex/detectron2
uv pip install -e . --no-build-isolation
cd ../..

# Install Detrex (Detection Transformers)
cd detrex
uv pip install -e . --no-build-isolation
cd ..

# Install Anomalib (for anomaly detection components)
cd anomalib
uv pip install -e . --no-build-isolation
cd ..
```

<!-- ### Verification

```python
import torch
import torchxrayvision as xrv
from src.memory_bank.anomaly_raddino import AnomalyRadDINO
from src.memory_bank.anomaly_raddino_v2 import AnomalyRadDINOv2

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Installation successful!")
``` -->

---

## 🚀 Quick Start

### 1. Download Pre-trained Models

**RadDINO Foundation Model** (ViT-B/16, already adapted from ViT-B/14):
```python
from src.utils import load_raddino

# Option A: Download converted checkpoint from Kaggle
# https://www.kaggle.com/models/thebeo182004/raddino-vitb14-topad16
raddino_ckpt_path = "path/to/rad_dino_vitb14_detrex_ready.pth"
raddino = load_raddino(raddino_ckpt_path)
```

**Anatomy Segmentation Model** (PSPNet, from torchxrayvision):
```python
from src.memory_bank.anomaly_detection import init_anatomy_segmentation_model

seg_model = init_anatomy_segmentation_model(device='cuda:0')
```

### 2. Build Memory Bank

```bash
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path "path/to/rad_dino_vitb14_detrex_ready.pth" \
  --train_normal_data_dir "path/to/normal_train_dir" \
  --save_path "path/to/raddino_memory_bank.pt" \
  --subsampling_ratio 0.05 \
  --batch_size 32 \
  --normal_num 1000 \
  --seed 0 \
  --use_aamb \
  --anatomy_dir "path/to/anatomy_mask_dir"
```

### 3. Run Inference with AAMB (AMBER v2 - with Anatomy-Aware Memory Banks)

```python
from src.memory_bank.binary_classification import binary_classification_inference_aamb

true_labels, anomaly_scores = binary_classification_inference_aamb(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/aamb_memory_bank.pt",
    normal_paths=["path/to/normal1.jpg", ...],
    abnormal_paths=["path/to/abnormal1.jpg", ...],
    num_neighbours=[1, 2, 4, 8, 10, 15, 20],
    batch_size=16,
    active_anatomies=['Lung', 'Heart']  # Optional: restrict to specific organs
)
```

---

## 📊 Datasets

AMBER is evaluated on two medical imaging benchmarks with standardized protocols:

### 1. **MedIAnomaly** Benchmark

Converts existing CXR datasets into few-shot anomaly detection protocols:

#### VinDr-CXR (MedIAnomaly)
- **Size**: ~6000 images (1024×1024), 4000 trainings and 2000 testings
- **Split**: N={1,5,10,20, 50, 100, 250, 500, 1000, 1500, 2000} normal references + validation/test abnormal samples
- **Kaggle**: [medanomaly_vindrcxr_1024](https://www.kaggle.com/datasets/thebeo182004/medanomaly-vindrcxr/data)
- **Anatomy Masks**: [MedIAnomaly-VinDrCXR-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-vindrcxr-anatomy-maps/notebook)

#### RSNA (MedIAnomaly)
- **Size**: ~5851 images (1024×1024), 3851 trainings and 2000 testings
- **Split**: N={1,5,10,20, 50+} normal references + validation/test pneumonia samples
- **Kaggle**: [MedIAnomaly-RSNA-1024](https://www.kaggle.com/datasets/thebeo182004/medianomaly-rsna/data)
- **Anatomy Masks**: [MedIAnomaly-RSNA-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-rsna-anatomy-maps)

### 2. **BMAD** Benchmark (Med-Anomalies)

- **Size**: ~26684 images (1024×1024), 8000 trainings, 1490 validations and 17194 testings
- **Protocol**: Few-shot protocol with standardized train/val/test splits
- **Kaggle**: [BMAD-RSNA-1024](https://www.kaggle.com/code/thebeo182004/bmad-rsna/output)
- **Anatomy Masks**: [BMAD-RSNA-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-rsna-anatomy-maps)

> **Note**: Datasets are extracted from original research ([MedIAnomaly](https://www.sciencedirect.com/science/article/abs/pii/S1361841525000489), [BMAD](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.pdf)) and hosted on Kaggle for convenient preprocessing and organization. They are identical to the original benchmark releases.

---

## 🏗️ Memory Bank Construction

### Overview

Memory banks store compressed normal patch features for k-NN based anomaly scoring. AMBER provides two variants:

| Variant | Features | Use Case |
|---------|----------|----------|
| **v1 (Standard)** | Single global memory bank | Baseline |
| **v2 (AAMB)** | 5 organ-specific sub-banks | Better anatomy-aware detection |

### Step-by-step Memory Bank Building

#### A. Prepare Data

```bash
# Directory structure required:
normal_train_dir/
├── img_001.jpg
├── img_002.jpg
└── ...

# For AAMB, also need anatomy masks:
anatomy_masks_dir/
├── img_001.npz  # Binary masks for 14 anatomies
├── img_002.npz
└── ...
```

**Generate anatomy masks** (see [Notebook 0.2](notebook/anomaly_detection/0.2%20torchxrayvision%20-%20Anatomy%20segmentation.ipynb)): You could use the available anatomy masks we provide on Kaggle (as we indicate above).


#### B. Build Memory Bank (AMBER v1)

```bash
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path "path/to/rad_dino_vitb14_detrex_ready.pth" \
  --train_normal_data_dir "path/to/normal_train_dir" \
  --save_path "path/to/raddino_memory_bank.pt" \
  --subsampling_ratio 0.05 \
  --batch_size 32 \
  --normal_num 1000 \
  --seed 0
```

**Parameters Explained**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--raddino_checkpoint_path` | **Required** | Path to **ViT-B/16 converted** RadDINO checkpoint (see `convert_rad_dino_for_detrex` in [src/utils.py](src/utils.py)) or pre-converted from [Kaggle](https://www.kaggle.com/models/thebeo182004/raddino-vitb14-topad16) |
| `--train_normal_data_dir` | **Required** | Directory containing healthy CXR reference images (PNG/JPG) |
| `--save_path` | `raddino_memory_bank.pt` | Output path for the compressed memory bank tensor |
| `--subsampling_ratio` | `0.05` | Coreset ratio: retain 5% of patches via K-Center Greedy. Lower = smaller bank, faster inference |
| `--batch_size` | `32` | Batch size for feature extraction. Adjust based on GPU VRAM |
| `--normal_num` | `1000` | Maximum number of normal images to use. Use all if `len(normal_dir) < normal_num` |
| `--seed` | `0` | Random seed for reproducibility |

#### C. Build Memory Bank with AAMB (AMBER v2)

```bash
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path "path/to/rad_dino_vitb14_detrex_ready.pth" \
  --train_normal_data_dir "path/to/normal_train_dir" \
  --anatomy_dir "path/to/anatomy_masks_dir" \
  --save_path "path/to/aamb_memory_bank.pt" \
  --subsampling_ratio 0.05 \
  --batch_size 16 \
  --use_aamb \
  --seed 0
```

**Additional AAMB Parameters**:

| Parameter | Description |
|-----------|-------------|
| `--use_aamb` | **Flag**: Enable Anatomy-Aware Memory Banking (5 separate sub-banks) |
| `--anatomy_dir` | **Required with `--use_aamb`**. Directory with `.npz` files containing binary anatomy masks. Mask order must match torchxrayvision definition (14 anatomies) |

**Output**: `aamb_memory_bank.pt` containing state_dict with 5 memory banks:
```python
torch.load("aamb_memory_bank.pt")
# Dict keys: 'memory_bank_0' (Clavicle), 'memory_bank_1' (Lung), 
#            'memory_bank_2' (Heart), 'memory_bank_3' (Diaphragm), 'memory_bank_4' (Mediastinum)
```

#### D. Understanding Coreset Compression

**K-Center Greedy** selects maximally distant subset of features:

```
Original bank size: N patches
Coreset ratio p: 0.05
Final bank size: 0.05 × N patches
```

**Trade-offs**:
- **p = 1.0** (100%): Full reference set, slower inference (~2000+ GFLOPs), larger memory
- **p = 0.05** (5%): Aggressive compression (~50-100 GFLOPs), fast inference
- **p = 0.1-0.2**: Recommended for good accuracy/speed balance

See [Notebook 3.0](notebook/anomaly_detection/3.0%20Computational%20complexity%20report.ipynb) for empirical complexity analysis.

---

## 🔍 Inference

### AMBER v1: Standard Global Memory Bank

```python
from src.memory_bank.binary_classification import binary_classification_inference

# Simple interface: returns true labels and anomaly scores
true_labels, predicted_scores = binary_classification_inference(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/raddino_memory_bank.pt",
    normal_paths=["img1.jpg", "img2.jpg", ...],
    abnormal_paths=["abnormal1.jpg", "abnormal2.jpg", ...],
    batch_size=16,
    num_neighbours=[1, 2, 4, 8, 10, 15, 20]
)

# Output structure:
# true_labels: (N,) binary array [0=normal, 1=abnormal]
# predicted_scores: dict[int -> dict[str -> array]]
#   Example: predicted_scores[4]['top10_percent'] -> (N,) anomaly scores for k=4, strategy='top10_percent'
```

### AMBER v2: Anatomy-Aware Memory Banking

```python
from src.memory_bank.binary_classification import binary_classification_inference_aamb

# AAMB adds anatomy-aware feature matching + online mask generation
true_labels, predicted_scores = binary_classification_inference_aamb(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/aamb_memory_bank.pt",
    normal_paths=["img1.jpg", "img2.jpg", ...],
    abnormal_paths=["abnormal1.jpg", "abnormal2.jpg", ...],
    batch_size=16,
    num_neighbours=[1, 2, 4, 8, 10, 15, 20],
    active_anatomies=['Lung', 'Heart']  # Optional: restrict to specific organs
)

# same output structure as v1
```

**Key Differences**:
- **v1**: Compares patches against all normal features (fast, less accurate)
- **v2**: Compares patches only against anatomically-matched normal features + generates anatomy masks online
  - Significantly lower
  - Better accuracy (anatomy-aware matching)
  - Optional: `active_anatomies` restricts to specific organs for focused analysis

### Multiple Aggregation Strategies

AMBER returns predictions for 7 different patch-to-image aggregation strategies:

```python
strategies = [
    "top1",              # Highest anomaly patch
    "top5",              # Mean of top 5 patches
    "top10",             # Mean of top 10 patches
    "top1_percent",      # Mean of top 1% patches
    "top5_percent",      # Mean of top 5% patches
    "top10_percent",     # Mean of top 10% patches (recommended)
    "softmax_weighted"   # Softmax-weighted mean (amplifies strong anomalies)
]

# Access scores for specific k and strategy
for k in num_neighbours:
    scores_k = predicted_scores[k]  # dict of strategy -> scores
    top10_scores = scores_k['top10_percent']  # (N,) anomaly scores
```

### Anomaly Map Visualization

```python
from src.memory_bank.utils import visualize_anomaly_map_aamb

result = visualize_anomaly_map_aamb(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/aamb_memory_bank.pt",
    image_path="path/to/cxr_image.jpg",
    annotations_path="path/to/annotations.json",  # COCO format with bounding boxes
    num_neighbours=4,
    aggregator_strategy='top10_percent',
    gamma=1.0,  # Gamma correction for heatmap
    threshold_method='percentile',
    threshold_param=99.0,
    alpha=0.45  # Overlay transparency
)

# Returns: figure, anomaly_map, threshold, filtered_mask, box_stats, image_score
```

---

## 📈 Evaluation & Analysis

### Quick Evaluation

```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Compute metrics for specific k and strategy
k, strategy = 4, 'top10_percent'
scores = predicted_scores[k][strategy]

auroc = roc_auc_score(true_labels, scores)
auprc = average_precision_score(true_labels, scores)

print(f"k={k}, strategy={strategy}")
print(f"AUROC: {auroc:.3f}")
print(f"AUPRC: {auprc:.3f}")
```

### Comprehensive Analysis

See [Evaluation Notebooks](#notebooks-guide) for:
- ROC/PR curve visualization
- Ablation studies (k, strategy impact)
- Baseline comparisons (reconstruction, self-supervised, memory-based)
- Failure case analysis
- Computational complexity reports

---

## 📓 Notebooks Guide

### Stage 0: Dataset & Foundation Model Exploration

| Notebook | Purpose | Key Takeaway |
|----------|---------|--------------|
| [0.0 AD Dataset Visualization](notebook/anomaly_detection/0.0%20AD%20dataset%20visualization.ipynb) | Visualize dataset statistics (normal vs abnormal splits across MedIAnomaly/BMAD) | Understand benchmark protocols & class imbalance |
| [0.1 torchxrayvision Overview](notebook/anomaly_detection/0.1%20torchxrayvision%20-%20Overview.ipynb) | Explore anatomy segmentation model from torchxrayvision | Verify mask quality before memory bank building |
| [0.2 Anatomy Segmentation](notebook/anomaly_detection/0.2%20torchxrayvision%20-%20Anatomy%20segmentation.ipynb) | Generate binary anatomy masks for all training images | Create `anatomy_masks_dir` for AAMB training |
| [0.3 VinDrCXR Anatomical Analysis](notebook/anomaly_detection/0.3%20VinDrCXR%20anatomical%20analysis.ipynb) | Compute Intersection-over-Area (IoA) between diseases and anatomies | Justify anatomical region selection (Lung, Heart, etc.) |

### Stage 1: Memory Bank Construction

| Notebook | Purpose | Command |
|----------|---------|---------|
| [1.0 AnomalyRadDINOv1 - Build Memory](notebook/anomaly_detection/1.0%20AnomalyRadDINOv1%20-%20Build%20memory.ipynb) | Build AMBER v1 (global bank) on different datasets | `python src/memory_bank/build_memory.py ...` |
| [1.1 AnomalyRadDINOv2 - Build Memory](notebook/anomaly_detection/1.1%20AnomalyRadDINOv2%20-%20Build%20memory.ipynb) | Build AMBER v2 (AAMB with sub-banks) on different datasets | `python src/memory_bank/build_memory.py --use_aamb ...` |

### Stage 2: Inference

| Notebook | Purpose | Difference |
|----------|---------|-----------|
| [2.0 AnomalyRadDINOv1 - Inference](notebook/anomaly_detection/2.0%20AnomalyRadDINOv1%20-%20Inference.ipynb) | Run inference on validation/test using v1 model | No anatomy masks, single global bank |
| [2.1 AnomalyRadDINOv2 - Inference (Full)](notebook/anomaly_detection/2.1%20AnomalyRadDINOv2%20-%20Inference%20(Full%20anats).ipynb) | Run inference using all 5 anatomy sub-banks | Online mask generation + anatomy-restricted matching |
| [2.2 AnomalyRadDINOv2 - Inference (Ablation)](notebook/anomaly_detection/2.2%20AnomalyRadDINOv2%20-%20Inference%20(Ablation%20studies).ipynb) | Ablation: use only specific anatomies (e.g., Lung only) | Isolate contribution of specific organs |

### Stage 3: Analysis & Results

| Notebook | Purpose | Output |
|----------|---------|--------|
| [3.0 Computational Complexity Report](notebook/anomaly_detection/3.0%20Computational%20complexity%20report.ipynb) | Profile FLOPs and GPU memory across configs | Left: GFLOPs comparison (v1 vs v2). Right: Peak VRAM curves |
| [3.1 MedIAnomaly VinDrCXR - Ablation](notebook/anomaly_detection/3.1%20MedIAnomaly%20VinDrCXR%20-%20Ablation%20study%20analysis.ipynb) | Systematic ablation: k, strategy, coreset ratio effects | AUROC/AUPRC grids for hyperparameter selection |
| [3.2 MedIAnomaly RSNA - Ablation](notebook/anomaly_detection/3.2%20MedIAnomaly%20RSNA%20-%20Ablation%20study%20analysis.ipynb) | Same as 3.1 but for RSNA dataset | Different distribution, benchmark-specific insights |
| [3.3 MedIAnomaly Benchmarking](notebook/anomaly_detection/3.3%20MedIAnomaly%20benchmarking%20against%20baselines.ipynb) | Compare AMBER v1 vs v2 vs reconstruction/SSL baselines | Summary table: AUROC/AUPRC across methods |
| [4.0 BMAD RSNA - Ablation](notebook/anomaly_detection/4.0%20BMAD%20RSNA%20-%20Ablation%20study%20analysis.ipynb) | Ablation on BMAD protocol (different eval protocol) | Validate generalization across protocols |
| [4.1 BMAD RSNA - Failure Analysis](notebook/anomaly_detection/4.1%20BMAD%20RSNA%20-%20Failure%20case%20analysis.ipynb) | Analyze false positives & false negatives | Identify failure modes & method limitations |

### Pre-computed Results

Inference results for all 5 runs are stored in [notebook/anomaly_detection/benchmarking/](notebook/anomaly_detection/benchmarking/).

<!-- ```
benchmarking/
├── medianomaly_vindrcxr/
│   ├── v1_n1_validation.npz
│   ├── v1_n1_test.npz
│   ├── v2_aamb_n1_validation.npz
│   └── v2_aamb_n1_test.npz
├── medianomaly_rsna/
│   └── ...
└── bmad_rsna/
    └── ...
``` -->
<!-- 
Load and verify results:

```python
import numpy as np

results = np.load("benchmarking/medianomaly_vindrcxr/v1_n1_test.npz")
true_labels = results['true_labels']  # (N,)
k1_top10_scores = results['k_1_top10_percent']  # (N,) anomaly scores for k=1, strategy='top10_percent'

print(f"Labels: {true_labels}")
print(f"Scores shape: {k1_top10_scores.shape}") -->
<!-- ``` -->

<!-- ---

## 📊 Results

### Few-shot Performance Summary

| Setting | Method | Backbone | AUROC (%) | AUPRC (%) |
|---------|--------|----------|-----------|-----------|
| MedIAnomaly VinDrCXR (N=1) | AMBER v1 | RadDINO ViT-B/16 | 71.2 ± 2.3 | 68.4 ± 3.1 |
| MedIAnomaly VinDrCXR (N=1) | **AMBER v2 (AAMB)** | RadDINO ViT-B/16 | **78.5 ± 1.8** | **76.9 ± 2.2** |
| MedIAnomaly VinDrCXR (N=20) | AMBER v1 | RadDINO ViT-B/16 | 85.3 ± 1.1 | 84.2 ± 1.4 |
| MedIAnomaly VinDrCXR (N=20) | **AMBER v2 (AAMB)** | RadDINO ViT-B/16 | **89.7 ± 0.9** | **88.6 ± 1.1** |
| MedIAnomaly RSNA (N=1) | AMBER v1 | RadDINO ViT-B/16 | 68.1 ± 2.8 | 64.3 ± 3.5 |
| MedIAnomaly RSNA (N=1) | **AMBER v2 (AAMB)** | RadDINO ViT-B/16 | **75.2 ± 2.1** | **71.8 ± 2.9** |
| BMAD RSNA | AMBER v1 | RadDINO ViT-B/16 | 82.4 ± 1.5 | 80.1 ± 1.8 |
| BMAD RSNA | **AMBER v2 (AAMB)** | RadDINO ViT-B/16 | **86.9 ± 1.2** | **85.3 ± 1.5** |

**Key Findings**:
- AAMB consistently improves v1 baseline by 5-10% AUROC across protocols
- Strongest gains in low-shot settings (N=1, 5) where anatomy awareness matters most
- Anatomy-specific retrieval reduces cross-anatomical semantic mismatch

See [Notebooks 3.1-4.1](#notebooks-guide) for detailed results, confidence intervals, and protocol-specific analysis.

--- -->

<!-- ## ⚡ Computational Complexity

### Memory & Inference Cost Analysis

See [Notebook 3.0](notebook/anomaly_detection/3.0%20Computational%20complexity%20report.ipynb) for empirical profiling.

**Peak GPU Memory** (Coreset Selection Phase):

```
Config: N=1000 normal images, p=0.05 coreset ratio
Method       Peak VRAM    Threshold (T4 GPU ~15GB)
─────────────────────────────────────────────────
AMBER v1:    ~8-10 GB     N ≤ 2000 before OOM
AMBER v2:    ~6-8 GB      N ≤ 2500 before OOM  (30% savings via region-wise compression)
```

**Inference Cost** (per test image):

```
Component          GFLOPs    Notes
──────────────────────────────────────────
RadDINO backbone   ~450      Frozen, same for v1/v2
K-NN (v1):         50-200    Varies with memory bank size
K-NN (v2):         40-160    ~20-30% less due to anatomy restriction
PSPNet masks:      +80       AAMB only, online generation
Total (v1):        500-650   GFLOPs per image
Total (v2):        610-840   GFLOPs per image
```

**Mitigation Strategies**:
- Use `--subsampling_ratio 0.05-0.1` for aggressive memory compression
- Multi-GPU inference via `concurrent.futures.ThreadPoolExecutor` (see Notebook 2.0)
- Precompute anatomy masks offline if evaluating many images with AAMB

---

## 🔄 Full Reproduction Pipeline

### Example: MedIAnomaly VinDrCXR (N=5)

```bash
# 1. Download dataset & anatomy masks from Kaggle
# Dataset: https://www.kaggle.com/datasets/thebeo182004/medanomaly-vindrcxr
# Anatomy: https://www.kaggle.com/code/thebeo182004/medianomaly-vindrcxr-anatomy-maps

# 2. Install AMBER
git clone ...
cd AMBER
uv sync
cd detrex/detectron2 && uv pip install -e . && cd ../..
cd detrex && uv pip install -e . && cd ..
cd anomalib && uv pip install -e . && cd ..

# 3. Build memory banks
# v1: Global bank
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path "path/to/rad_dino_vitb14_detrex_ready.pth" \
  --train_normal_data_dir "path/to/medianomaly_vindrcxr/train_normal" \
  --save_path "memory_banks/vindrcxr_v1_n5.pt" \
  --normal_num 5 \
  --subsampling_ratio 0.1

# v2: AAMB with sub-banks
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path "path/to/rad_dino_vitb14_detrex_ready.pth" \
  --train_normal_data_dir "path/to/medianomaly_vindrcxr/train_normal" \
  --anatomy_dir "path/to/anatomy_masks/train_normal" \
  --save_path "memory_banks/vindrcxr_v2_aamb_n5.pt" \
  --normal_num 5 \
  --subsampling_ratio 0.1 \
  --use_aamb

# 4. Run inference (see Notebooks 2.0-2.2 for detailed examples)
# Results → benchmarking/ directory

# 5. Evaluate & visualize (see Notebooks 3.1-4.1)
```

---

## 🎓 Understanding the Architecture

### Memory Bank Organization

```
AMBER v1 (Global Bank):
┌─────────────────────────────┐
│  Normal Reference Features  │  (all patches from all anatomies)
│  [Z_1, Z_2, ..., Z_M]       │  Size: M patches × 768 dim
│  ↓ K-Center Greedy (p=0.1)  │
│  [Z_c1, Z_c2, ..., Z_ck]    │  Compressed: 0.1M patches × 768 dim
└─────────────────────────────┘

Query patch (e.g., from lung):
  → Compute k-NN distances to ALL patches
  → Highest distance = anomaly score
  Problem: May match against heart/mediastinum features ✗

───────────────────────────────────────────────────────────────

AMBER v2 with AAMB (Anatomy-Aware Sub-Banks):
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Clavicle    │ │    Lung      │ │    Heart     │ │  Diaphragm   │ │  Mediastinum     │
│    Bank      │ │    Bank      │ │    Bank      │ │    Bank      │ │    Bank          │
├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────────┤
│ [Z_c1, ...] │ │ [Z_l1, ...] │ │ [Z_h1, ...] │ │ [Z_d1, ...] │ │ [Z_m1, ...]      │
│ 1.2K patch  │ │ 9.5K patch  │ │ 2.3K patch  │ │ 4.1K patch  │ │ 1.8K patch       │
│     ↓        │ │     ↓        │ │     ↓        │ │     ↓        │ │     ↓            │
│ 120 core    │ │ 950 core    │ │ 230 core    │ │ 410 core    │ │ 180 core         │
│(p=0.1)      │ │ (p=0.1)     │ │ (p=0.1)     │ │ (p=0.1)     │ │ (p=0.1)          │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘

Query patch (e.g., from lung):
  1. Generate anatomy mask → patch belongs to Lung ✓
  2. Compute k-NN only against Lung Bank
  3. Only matches valid anatomical references ✓
  Result: Better anomaly scores, ~30% less memory, ~20-30% fewer FLOPs
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: RadDINO checkpoint incompatible with Detrex**

```
Error: Expected ViT-B/16 but got ViT-B/14
```

**Solution**: Use converted checkpoint from [Kaggle](https://www.kaggle.com/models/thebeo182004/raddino-vitb14-topad16) or convert locally:

```python
from src.utils import convert_rad_dino_for_detrex

convert_rad_dino_for_detrex(
    backbone_safetensor_path="microsoft_rad_dino_vitb14.safetensors",
    output_pth="rad_dino_vitb14_detrex_ready.pth"
)
```

**Issue 2: OOM during coreset selection with large N**

```
RuntimeError: CUDA out of memory
```

**Solution**: Use smaller `--subsampling_ratio` or reduce `--batch_size`:

```bash
# Instead of: --subsampling_ratio 0.1 --batch_size 32
# Use:
python src/memory_bank/build_memory.py \
  ... \
  --subsampling_ratio 0.05 \
  --batch_size 8
```

**Issue 3: Multi-GPU inference fails**

```
CUDA device mismatch or ThreadPoolExecutor error
```

**Solution**: Ensure both GPUs are visible and manually handle synchronization:

```bash
export CUDA_VISIBLE_DEVICES=0,1

python -c "
import torch
print(f'Visible GPUs: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'GPU 1: {torch.cuda.get_device_name(1)}')
"
```

---

## 📚 Citation

If you use AMBER in your research, please cite:

```bibtex
@article{amber2024,
  title={AMBER: Anatomy-aware Memory Banks for Efficient Radiographic Anomaly Detection},
  author={Nguyen, Huu The and ...},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! For major changes, please open an issue first to discuss proposed changes.

---

## 📞 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

---

## Acknowledgments

- **RadDINO**: Microsoft's medical foundation model ([Hugging Face](https://huggingface.co/microsoft/rad-dino))
- **Datasets**: [MedIAnomaly](https://www.sciencedirect.com/science/article/abs/pii/S1361841525000489), [BMAD](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.pdf)
- **Detrex & Detectron2**: Facebook Research (now Meta)

---

**Last Updated**: July 2026  
**Version**: 1.0 -->
