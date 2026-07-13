# AMBER: Anatomy-aware Memory Banks for Efficient Radiographic Anomaly Detection

AMBER is the publication name of the project previously referred to in code and notebooks as AnomalyRadDINO. The two implementation variants are:

- AMBER v1: global memory bank, without anatomy-aware memory partitioning
- AMBER v2: AMBER with AAMB, where AAMB stands for Anatomy-Aware Memory Banking

The goal of this repository is reproducibility. The code, notebooks, datasets, and checkpoints are organized so readers can rebuild the memory banks, run inference, reproduce the ablations, and verify the benchmarking results reported in the scientific report.
## Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [Environment Setup](#environment-setup)
4. [Data and Pretrained Assets](#data-and-pretrained-assets)
5. [Reproduction Workflow](#reproduction-workflow)
6. [Memory Bank Construction](#memory-bank-construction)
7. [Inference](#inference)
8. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
9. [Notebooks Guide](#notebooks-guide)
10. [Computational Complexity](#computational-complexity)
11. [Citation](#citation)

## Overview

AMBER is a few-shot unsupervised anomaly detection framework for chest X-ray imaging. It uses a frozen radiology foundation model, RadDINO, to extract patch-level features from normal reference images, stores those features in a memory bank, and scores test images by nearest-neighbor retrieval.

The key idea is anatomical structure. Instead of one global memory bank, AMBER v2 splits the normal reference features into organ-specific sub-banks and restricts retrieval to the anatomies that correspond to the query patch. This reduces cross-anatomical semantic mismatch, which is especially important in CXR where lung, heart, diaphragm, mediastinum, and clavicle regions have very different normal appearances.

The repository contains both the implementation and the reproducibility material needed to follow the paper end to end:

- source code in [src/](src)
- step-by-step notebooks in [notebook/anomaly_detection/](notebook/anomaly_detection)
- benchmark results under [notebook/anomaly_detection/benchmarking/](notebook/anomaly_detection/benchmarking)

## Repository Layout

- [src/utils.py](src/utils.py): RadDINO checkpoint conversion, preprocessing transforms, and annotation helpers
- [src/memory_bank/build_memory.py](src/memory_bank/build_memory.py): memory bank construction for AMBER v1 and v2
- [src/memory_bank/anomaly_detection.py](src/memory_bank/anomaly_detection.py): model initialization and single-image inference helpers
- [src/memory_bank/binary_classification.py](src/memory_bank/binary_classification.py): dataset-level inference for normal vs abnormal evaluation
- [src/memory_bank/anomaly_raddino.py](src/memory_bank/anomaly_raddino.py): AMBER v1 model definition
- [src/memory_bank/anomaly_raddino_v2.py](src/memory_bank/anomaly_raddino_v2.py): AMBER v2 model definition with AAMB
- [src/memory_bank/cxr_dataset.py](src/memory_bank/cxr_dataset.py): normal-image dataset for memory construction
- [src/memory_bank/binary_cxr_dataset.py](src/memory_bank/binary_cxr_dataset.py): paired normal/abnormal dataset for inference
- [src/memory_bank/utils.py](src/memory_bank/utils.py): anatomy merging, mask loading, visualization, and analysis utilities
- [notebook/anomaly_detection/](notebook/anomaly_detection): notebooks for data exploration, memory construction, inference, ablations, benchmarking, complexity, and failure analysis
- [detrex/](detrex) and [anomalib/](anomalib): vendored submodules required by the implementation

## Environment Setup

AMBER is developed for Python 3.12+ with PyTorch 2.13+ and CUDA 11.8+ recommended.

### 1. Clone the repository

```bash
git clone https://github.com/uet-dis/AMBER.git
git submodule update --init --recursive
```

### 2. Install dependencies

Use `uv` for dependency management:

```bash
pip install uv
uv sync
```

The repository also depends on the local `detrex` and `anomalib` submodules. Install them in this order:

```bash
cd detrex/detectron2
uv pip install -e . --no-build-isolation
cd ../..

cd detrex
uv pip install -e . --no-build-isolation
cd ..

cd anomalib
uv pip install -e . --no-build-isolation
cd ..
```

### 3. Optional verification

If you want a quick sanity check, run a small import test after installation:

```python
import torch
from src.utils import load_raddino

print(torch.__version__)
print(torch.cuda.is_available())
```

## Data and Pretrained Assets

AMBER is evaluated on the following benchmarks:

- [MedIAnomaly VinDrCXR 1024 x 1024](https://www.kaggle.com/datasets/thebeo182004/medanomaly-vindrcxr/data)
- [MedIAnomaly RSNA 1024 x 1024](https://www.kaggle.com/datasets/thebeo182004/medianomaly-rsna/data)
- [BMAD-RSNA-1024](https://www.kaggle.com/code/thebeo182004/bmad-rsna/output)

The Kaggle copies of these datasets are mirrors of the original benchmark releases used in the the following papers [MedIAnomaly](https://www.sciencedirect.com/science/article/abs/pii/S1361841525000489), [BMAD](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.pdf). They are provided for convenience so readers can reproduce preprocessing and experiment organization more easily.

The corresponding anatomy-mask resources used for AMBER v2 are:

- [MedIAnomaly-VinDrCXR-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-vindrcxr-anatomy-maps/notebook)
- [MedIAnomaly-RSNA-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-rsna-anatomy-maps)
- [BMAD-RSNA-Anatomy-Maps](https://www.kaggle.com/code/thebeo182004/medianomaly-rsna-anatomy-maps)

These masks follow the torchxrayvision anatomy convention and are required for AAMB training. For the dataset setup and anatomical analysis workflow, see [notebook/anomaly_detection/0.0 AD dataset visualization.ipynb](notebook/anomaly_detection/0.0%20AD%20dataset%20visualization.ipynb), [notebook/anomaly_detection/0.1 torchxrayvision - Overview.ipynb](notebook/anomaly_detection/0.1%20torchxrayvision%20-%20Overview.ipynb), and [notebook/anomaly_detection/0.3 VinDrCXR anatomical analysis.ipynb](notebook/anomaly_detection/0.3%20VinDrCXR%20anatomical%20analysis.ipynb).

### RadDINO checkpoint

AMBER uses RadDINO as the frozen feature extractor. The public model is available as [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) on Hugging Face, but the original architecture is ViT-B/14 and was pretrained at 518 x 518 resolution.

Because the CXR benchmarks in this project use 1024 x 1024 images and the implementation expects a ViT-B/16 compatible backbone, the checkpoint must be converted before use. There are two supported options:

1. Convert locally with [src/utils.py](src/utils.py), using `convert_rad_dino_for_detrex`
2. Use the authors' pre-converted checkpoint distributed on Kaggle: [RadDINO_ViTB14_ToPad16](https://www.kaggle.com/models/thebeo182004/raddino-vitb14-topad16)

In both cases, the checkpoint passed to `--raddino_checkpoint_path` must be the converted ViT-B/16 version.

## Reproduction Workflow

The recommended workflow is:

1. Prepare the environment and install dependencies
2. Select a benchmark and download the image data and anatomy masks
3. Build the memory bank with AMBER v1 or AMBER v2
4. Run inference on the validation or test split
5. Reproduce the benchmark tables, ablations, and failure analysis from the notebooks

The notebooks provide the most direct guided path through the implementation. The corresponding scripts in [src/](src) are the executable version of the same workflow.

## Memory Bank Construction

The memory-bank pipeline is implemented in [src/memory_bank/build_memory.py](src/memory_bank/build_memory.py). It builds a reference bank from healthy samples and optionally splits the bank into anatomy-specific sub-banks for AAMB.

### Command line usage

AMBER v1, global memory bank:

```bash
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path path/to/rad_dino_vitb14_detrex_ready.pth \
  --train_normal_data_dir path/to/normal_train_dir \
  --save_path path/to/raddino_memory_bank.pt \
  --subsampling_ratio 0.05 \
  --batch_size 32 \
  --normal_num 1000 \
  --seed 0
```

AMBER v2, anatomy-aware memory banks:

```bash
python src/memory_bank/build_memory.py \
  --raddino_checkpoint_path path/to/rad_dino_vitb14_detrex_ready.pth \
  --train_normal_data_dir path/to/normal_train_dir \
  --anatomy_dir path/to/anatomy_mask_dir \
  --save_path path/to/aamb_memory_bank.pt \
  --subsampling_ratio 0.05 \
  --batch_size 16 \
  --use_aamb \
  --seed 0
```

### Parameters explained

`--raddino_checkpoint_path`
: Path to the converted RadDINO checkpoint in ViT-B/16 format. This is required for both v1 and v2.

`--train_normal_data_dir`
: Directory containing healthy reference images used to build the memory bank. This directory should contain the normal training split only.

`--anatomy_dir`
: Directory containing offline anatomy-mask files in `.npz` format. This is required only when `--use_aamb` is enabled. Each mask file must have the same stem as the corresponding image file, and the mask channel order must follow the torchxrayvision anatomy definition.

`--save_path`
: Output path for the constructed memory bank.

`--subsampling_ratio`
: Coreset ratio used by K-Center Greedy. Smaller values reduce memory and inference cost, but can reduce retrieval coverage.

`--batch_size`
: Batch size used during feature extraction.

`--normal_num`
: Maximum number of normal reference images to use when building the memory bank.

`--seed`
: Random seed for reproducible subset selection.

`--use_aamb`
: Enables AMBER v2. When this flag is set, the model uses anatomy-aware sub-banks rather than a single global bank.

### Memory bank outputs

AMBER v1 saves a single tensor to `raddino_memory_bank.pt`.

AMBER v2 saves a dictionary containing one memory bank per selected anatomy. The keys are:

- `memory_bank_0`: Clavicle
- `memory_bank_1`: Lung
- `memory_bank_2`: Heart
- `memory_bank_3`: Facies Diaphragmatica
- `memory_bank_4`: Mediastinum

### Recommended notebooks

- [notebook/anomaly_detection/1.0 AnomalyRadDINOv1 - Build memory.ipynb](notebook/anomaly_detection/1.0%20AnomalyRadDINOv1%20-%20Build%20memory.ipynb)
- [notebook/anomaly_detection/1.1 AnomalyRadDINOv2 - Build memory.ipynb](notebook/anomaly_detection/1.1%20AnomalyRadDINOv2%20-%20Build%20memory.ipynb)

Those notebooks show the same pipeline as the script, but with commentary and intermediate visual checks.

## Inference

The inference utilities are implemented in [src/memory_bank/binary_classification.py](src/memory_bank/binary_classification.py) and [src/memory_bank/anomaly_detection.py](src/memory_bank/anomaly_detection.py).

### AMBER v1 inference

Use `binary_classification_inference` for the global memory-bank variant:

```python
from src.memory_bank.binary_classification import binary_classification_inference

true_labels, predicted_scores = binary_classification_inference(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/raddino_memory_bank.pt",
    normal_paths=["img1.jpg", "img2.jpg"],
    abnormal_paths=["abnormal1.jpg", "abnormal2.jpg"],
    batch_size=16,
    num_neighbours=[1, 2, 4, 8, 10, 15, 20],
)
```

### AMBER v2 inference

Use `binary_classification_inference_aamb` for the anatomy-aware variant:

```python
from src.memory_bank.binary_classification import binary_classification_inference_aamb

true_labels, predicted_scores = binary_classification_inference_aamb(
    raddino_checkpoint_path="path/to/rad_dino_vitb14_detrex_ready.pth",
    memory_bank_path="path/to/aamb_memory_bank.pt",
    normal_paths=["img1.jpg", "img2.jpg"],
    abnormal_paths=["abnormal1.jpg", "abnormal2.jpg"],
    batch_size=16,
    num_neighbours=[1, 2, 4, 8, 10, 15, 20],
    active_anatomies=["Lung", "Heart"],
)
```

### What the inference functions return

The dataset-level inference APIs return:

- `true_labels`: binary ground-truth labels, where 0 means normal and 1 means abnormal
- `predicted_scores`: a nested dictionary indexed by `k` and aggregation strategy

For each `k`, the available aggregation strategies are:

- `top1`
- `top5`
- `top10`
- `top1_percent`
- `top5_percent`
- `top10_percent`
- `softmax_weighted`

The single-image helper in [src/memory_bank/anomaly_detection.py](src/memory_bank/anomaly_detection.py) returns the anomaly map and the image-level anomaly score. For publication figures, use [src/memory_bank/utils.py](src/memory_bank/utils.py) and [notebook/anomaly_detection/2.1 AnomalyRadDINOv2 - Inference (Full anats).ipynb](notebook/anomaly_detection/2.1%20AnomalyRadDINOv2%20-%20Inference%20(Full%20anats).ipynb).

### Important inference note for AAMB

AMBER v2 generates anatomy masks online during inference using the segmentation model from torchxrayvision. This means the inference FLOPs include the segmentation pass, and there is no need to pre-store anatomy masks on disk for inference-time execution.

### Recommended notebooks

- [notebook/anomaly_detection/2.0 AnomalyRadDINOv1 - Inference.ipynb](notebook/anomaly_detection/2.0%20AnomalyRadDINOv1%20-%20Inference.ipynb)
- [notebook/anomaly_detection/2.1 AnomalyRadDINOv2 - Inference (Full anats).ipynb](notebook/anomaly_detection/2.1%20AnomalyRadDINOv2%20-%20Inference%20(Full%20anats).ipynb)
- [notebook/anomaly_detection/2.2 AnomalyRadDINOv2 - Inference (Ablation studies).ipynb](notebook/anomaly_detection/2.2%20AnomalyRadDINOv2%20-%20Inference%20(Ablation%20studies).ipynb)

## Evaluation and Benchmarking

The repository includes notebooks for reproducing the paper's analysis stage:

- [notebook/anomaly_detection/3.1 MedIAnomaly VinDrCXR - Ablation study analysis.ipynb](notebook/anomaly_detection/3.1%20MedIAnomaly%20VinDrCXR%20-%20Ablation%20study%20analysis.ipynb)
- [notebook/anomaly_detection/3.2 MedIAnomaly RSNA - Ablation study analysis.ipynb](notebook/anomaly_detection/3.2%20MedIAnomaly%20RSNA%20-%20Ablation%20study%20analysis.ipynb)
- [notebook/anomaly_detection/3.3 MedIAnomaly benchmarking against baselines.ipynb](notebook/anomaly_detection/3.3%20MedIAnomaly%20benchmarking%20against%20baselines.ipynb)
- [notebook/anomaly_detection/4.0 BMAD RSNA - Ablation study analysis.ipynb](notebook/anomaly_detection/4.0%20BMAD%20RSNA%20-%20Ablation%20study%20analysis.ipynb)
- [notebook/anomaly_detection/4.1 BMAD RSNA - Failure case analysis.ipynb](notebook/anomaly_detection/4.1%20BMAD%20RSNA%20-%20Failure%20case%20analysis.ipynb)

The benchmarking outputs used in the paper are stored in [notebook/anomaly_detection/benchmarking/](notebook/anomaly_detection/benchmarking/). Readers can load those files directly to verify the reported results or reproduce the tables and plots without rerunning every experiment from scratch.

### Typical analysis tasks

- AUROC and AUPRC computation for each `k` and aggregation strategy
- ROC and precision-recall curve plotting
- memory-bank and coreset-ratio ablations
- comparison of AMBER v1 vs AMBER v2
- failure-case inspection for false positives and false negatives

## Notebooks Guide

### Stage 0: Data exploration and anatomy analysis

- [0.0 AD dataset visualization.ipynb](notebook/anomaly_detection/0.0%20AD%20dataset%20visualization.ipynb): dataset distribution across MedIAnomaly and BMAD
- [0.1 torchxrayvision - Overview.ipynb](notebook/anomaly_detection/0.1%20torchxrayvision%20-%20Overview.ipynb): small experiment with the anatomy segmentation model
- [0.3 VinDrCXR anatomical analysis.ipynb](notebook/anomaly_detection/0.3%20VinDrCXR%20anatomical%20analysis.ipynb): IoA analysis between anatomies and diseases, plus anatomy area statistics and supporting plots

### Stage 1: Memory construction

- [1.0 AnomalyRadDINOv1 - Build memory.ipynb](notebook/anomaly_detection/1.0%20AnomalyRadDINOv1%20-%20Build%20memory.ipynb)
- [1.1 AnomalyRadDINOv2 - Build memory.ipynb](notebook/anomaly_detection/1.1%20AnomalyRadDINOv2%20-%20Build%20memory.ipynb)

### Stage 2: Inference

- [2.0 AnomalyRadDINOv1 - Inference.ipynb](notebook/anomaly_detection/2.0%20AnomalyRadDINOv1%20-%20Inference.ipynb)
- [2.1 AnomalyRadDINOv2 - Inference (Full anats).ipynb](notebook/anomaly_detection/2.1%20AnomalyRadDINOv2%20-%20Inference%20(Full%20anats).ipynb)
- [2.2 AnomalyRadDINOv2 - Inference (Ablation studies).ipynb](notebook/anomaly_detection/2.2%20AnomalyRadDINOv2%20-%20Inference%20(Ablation%20studies).ipynb)

### Stage 3: Analysis and results

- [3.0 Computational complexity report.ipynb](notebook/anomaly_detection/3.0%20Computational%20complexity%20report.ipynb)
- [3.1 MedIAnomaly VinDrCXR - Ablation study analysis.ipynb](notebook/anomaly_detection/3.1%20MedIAnomaly%20VinDrCXR%20-%20Ablation%20study%20analysis.ipynb)
- [3.2 MedIAnomaly RSNA - Ablation study analysis.ipynb](notebook/anomaly_detection/3.2%20MedIAnomaly%20RSNA%20-%20Ablation%20study%20analysis.ipynb)
- [3.3 MedIAnomaly benchmarking against baselines.ipynb](notebook/anomaly_detection/3.3%20MedIAnomaly%20benchmarking%20against%20baselines.ipynb)
- [4.0 BMAD RSNA - Ablation study analysis.ipynb](notebook/anomaly_detection/4.0%20BMAD%20RSNA%20-%20Ablation%20study%20analysis.ipynb)
- [4.1 BMAD RSNA - Failure case analysis.ipynb](notebook/anomaly_detection/4.1%20BMAD%20RSNA%20-%20Failure%20case%20analysis.ipynb)

## Computational Complexity

The complexity notebook [notebook/anomaly_detection/3.0 Computational complexity report.ipynb](notebook/anomaly_detection/3.0%20Computational%20complexity%20report.ipynb) profiles the cost of the v1 and v2 pipelines.

For AMBER v1, the main cost comes from RadDINO feature extraction plus nearest-neighbor retrieval over the global memory bank.

For AMBER v2, the inference cost includes the same backbone and retrieval stages, plus the online anatomy-segmentation step used to generate masks in-flight. This means v2 improves anatomical consistency but does add extra FLOPs at inference time.

When comparing methods, keep the following in mind:

- memory-bank coreset selection reduces search cost as the subsampling ratio decreases
- AAMB reduces anatomical mismatch but introduces segmentation overhead
- the reported complexity should always be interpreted together with the chosen `k` values and aggregation strategy

## Citation

If you use AMBER in your research, please cite our paper. A BibTeX entry will be added in the final release version of the manuscript.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For issues, questions, or reproducibility questions, open a GitHub issue in this repository.