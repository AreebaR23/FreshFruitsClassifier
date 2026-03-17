# FreshFruitsClassifier: Food Freshness and Quality Classification

## Project Overview

This project trains deep learning models to classify fruit images as **fresh** or **spoiled** using convolutional neural networks (CNNs). It includes:

- A **baseline CNN** model built from scratch.
- **Transfer learning** models based on **ResNet18** and **EfficientNet-B0**.
- A **data preparation pipeline** that organizes Kaggle datasets into standard `train/`, `val/`, and `test/` splits.
- **Evaluation utilities** for computing metrics and generating visualizations (confusion matrices, training curves, prediction grids).

The goal is for anyone to be able to **download the data, preprocess it, train the models, and reproduce the main results** using only the instructions in this repository.

---

## Repository Structure

```text
FreshFruitsClassifier/
├── data/                   # Datasets (raw + processed)
│   ├── raw/                # Original Kaggle downloads
│   ├── processed/          # Preprocessed train/val/test splits
│   └── README.md           # Dataset download and setup instructions
├── models/                 # Saved model checkpoints
├── notebooks/              # Jupyter notebooks for exploration/demo
├── results/                # Evaluation outputs and visualizations
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # Model architectures (baseline, ResNet, EfficientNet)
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation and plotting utilities
│   ├── utils.py            # Helper functions (checkpointing, metrics, etc.)
│   └── prepare_data.py     # Dataset preparation script
├── requirements.txt        # Python dependencies
├── report.md               # Project report (detailed write-up)
└── README.md               # This file
```

---

## Setup Instructions

### 1. Create and activate a virtual environment

From the project root (`FreshFruitsClassifier/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, matplotlib, seaborn, and any other libraries required by the code.

---

## Dataset: Download and Preprocessing

All dataset-related instructions are also summarized in `data/README.md`.

### 1. Download datasets from Kaggle

We use two public datasets:

1. **Fruits Fresh and Rotten for Classification**  
   URL: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

2. **Fruits-360**  
   URL: https://www.kaggle.com/datasets/moltean/fruits

First, install and configure the Kaggle CLI (once):

```bash
pip install kaggle
```

Set up your Kaggle API credentials (follow steps on https://www.kaggle.com/account), then in the `data/` directory run:

```bash
cd data

kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
kaggle datasets download -d moltean/fruits

unzip fruits-fresh-and-rotten-for-classification.zip -d raw/
unzip fruits.zip -d raw/

cd ..
```

### 2. Preprocess the data

From the project root:

```bash
cd src

python prepare_data.py \
    --raw_dir ../data/raw \
    --processed_dir ../data/processed \
    --max_samples 200

cd ..
```

This script:

- Reads the raw Kaggle datasets from `data/raw/`.
- Creates a standardized directory structure in `data/processed/`:
  - `train/fresh/`, `train/spoiled/`
  - `val/fresh/`, `val/spoiled/`
  - `test/fresh/`, `test/spoiled/`
- Optionally subsamples to at most `--max_samples` images per class to keep experiments fast.

### 3. Small Sample Dataset (Optional)

If you prefer a minimal dataset for quick tests:

- After running `prepare_data.py`, you can manually copy a handful of images (e.g., 10–20 per class) from `data/processed/train/` into a separate folder (e.g., `data/sample/`) and point `--raw_dir` or custom scripts there.
- Alternatively, reduce `--max_samples` (e.g., `--max_samples 50`) to create a very small processed dataset for rapid prototyping.

The repository itself does not include the full Kaggle datasets (due to size and licensing), but the instructions above allow you to reconstruct the exact splits.

---

## How to Train the Models

All training is performed via `src/train.py`. Make sure your virtual environment is active and data is preprocessed.

From the project root:

```bash
cd src
```

### 1. Train the baseline CNN (lightweight)

```bash
python train.py \
    --model baseline \
    --lightweight \
    --image_size 128 \
    --batch_size 16 \
    --epochs 8 \
    --num_workers 0 \
    --device cpu \
    --reuse_if_exists
```

### 2. Train the baseline CNN (full, smaller images)

```bash
python train.py \
    --model baseline \
    --image_size 96 \
    --batch_size 16 \
    --epochs 8 \
    --reuse_if_exists
```

### 3. Train ResNet18 (full fine-tuning)

```bash
python train.py \
    --model resnet18 \
    --image_size 128 \
    --batch_size 16 \
    --epochs 5 \
    --num_workers 0 \
    --no_augment \
    --reuse_if_exists
```

### 4. Train ResNet18 (frozen backbone)

```bash
python train.py \
    --model resnet18 \
    --freeze_backbone \
    --image_size 128 \
    --batch_size 16 \
    --epochs 5 \
    --num_workers 0 \
    --no_augment \
    --reuse_if_exists
```

### 5. Train baseline without augmentation

```bash
python train.py \
    --model baseline \
    --no_augment \
    --batch_size 16 \
    --epochs 5 \
    --num_workers 0 \
    --reuse_if_exists
```

### 6. Train EfficientNet-B0 (transfer learning)

```bash
python train.py \
    --model efficientnet_b0 \
    --image_size 128 \
    --batch_size 16 \
    --epochs 5 \
    --num_workers 0 \
    --reuse_if_exists
```

Each run will save model checkpoints and training history in the `models/` directory (e.g., `baseline_best.pth`, `resnet18_best.pth`, `efficientnet_b0_best.pth`, and corresponding `*_history.pth` files).

---

## How to Evaluate the Models

Evaluation and result visualization are handled by `src/evaluate.py`. Make sure you have trained (or downloaded) the corresponding checkpoints in `models/`.

From `src/`:

### 1. Evaluate the baseline model

```bash
python evaluate.py \
    --model baseline \
    --data_dir ../data/processed \
    --checkpoint ../models/baseline_best.pth \
    --results_dir ../results \
    --device cpu
```

### 2. Evaluate baseline with prediction grid

```bash
python evaluate.py \
    --model baseline \
    --data_dir ../data/processed \
    --checkpoint ../models/baseline_best.pth \
    --results_dir ../results \
    --device cpu \
    --save_prediction_grid \
    --num_grid_images 16
```

### 3. Evaluate ResNet18

```bash
python evaluate.py \
    --model resnet18 \
    --data_dir ../data/processed \
    --checkpoint ../models/resnet18_best.pth \
    --results_dir ../results \
    --device cpu \
    --save_prediction_grid \
    --num_grid_images 16
```

### 4. Evaluate EfficientNet-B0

```bash
python evaluate.py \
    --model efficientnet_b0 \
    --data_dir ../data/processed \
    --checkpoint ../models/efficientnet_b0_best.pth \
    --results_dir ../results \
    --device cpu \
    --save_prediction_grid \
    --num_grid_images 16
```

These commands will:

- Compute quantitative metrics (accuracy, confusion matrix, etc.).
- Save visualizations (confusion matrices, prediction grids, and optionally training curves) into the `results/` directory.

---

## Expected Outputs

After following the steps above, you should have:

- **Processed dataset** under `data/processed/` with `train/`, `val/`, `test/` splits.
- **Trained models** in `models/`:
  - `baseline_best.pth`, `baseline_final.pth`, `baseline_history.pth`
  - `resnet18_best.pth`, `resnet18_final.pth`, `resnet18_history.pth`
  - `efficientnet_b0_best.pth`, `efficientnet_b0_final.pth`, `efficientnet_b0_history.pth`
- **Evaluation results** in `results/`:
  - Metric summaries (accuracy, confusion matrix values).
  - Confusion matrix plots.
  - Prediction grid images showing predictions on sample test images.
  - Training/validation loss and accuracy curves (when plotted using `plot_training_history`).

---

## Reproducing the Results 

1. **Clone the repository and set up environment**
   - Create and activate a virtualenv.
   - Install dependencies from `requirements.txt`.

2. **Download and preprocess data**
   - Use Kaggle CLI to download the two datasets into `data/`.
   - Unzip into `data/raw/`.
   - Run `src/prepare_data.py` to create `data/processed/`.

3. **Train models**
   - Run one or more `python train.py` commands as listed above.

4. **Evaluate models**
   - Run `python evaluate.py` for baseline, ResNet18, and EfficientNet-B0.

5. **Inspect outputs**
   - Open figures in `results/` (confusion matrices, prediction grids, curves).
   - Optionally open `report.md` for a detailed write-up.