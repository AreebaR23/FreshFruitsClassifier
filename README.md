# FreshFruitsClassifier : Food Freshness and Quality Classification

In this project we aim to train a convolutional neural network that can accuractely classify fruit images as fresh or spoiled. As a possible addition to this project we plan to add a model that predicts the shelf life of a fruit given the image of that fruit.


## Project Overview

**Aim:** Build a CNN-based image classification system that detects whether a food item is fresh or spoiled using deep learning.

**Key Features:**
- Baseline CNN model built from scratch
- Pretrained model fine-tuning (ResNet, EfficientNet)
- Comprehensive data augmentation pipeline
- Detailed evaluation metrics and visualization
- Model comparison and analysis

##  Project Structure

```
FreshSense/
├── data/                   # Dataset storage
│   ├── raw/               # Original Kaggle datasets
│   ├── processed/         # Organized train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── README.md
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models.py         # Model architectures
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation utilities
│   ├── utils.py          # Helper functions
│   └── prepare_data.py   # Dataset preparation
├── results/              # Experiment results and visualizations
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle account for dataset download

### Installation

# 1. Clone the repository:
```bash
git clone <repository-url>
cd freshsense
```
# 2. python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

6. Set up Kaggle API:
```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json in ~/.kaggle/
# Get it from: https://www.kaggle.com/account
```

##  Datasets

We will be using two Kaggle datasets:

1. **Fruits Fresh and Rotten for Classification**
   - URL: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

2. **Fruits Dataset**
   - URL: https://www.kaggle.com/datasets/moltean/fruits

### Download and Prepare the data

```bash
cd data

# Download the datasets
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
kaggle datasets download -d moltean/fruits

# Extract to raw/ directory
unzip fruits-fresh-and-rotten-for-classification.zip -d raw/
unzip fruits.zip -d raw/

# Organize into train/val/test splits
cd ../src

# Prepare small dataset (300 samples/class)
python prepare_data.py --max_samples 300


## Training

### Train Baseline CNN

```bash
python train.py --model baseline --lightweight --batch_size 16 --epochs 15 --device cpu

```

### Train with Pretrained ResNet50

```bash
python train.py --model resnet50 --batch_size 32 --epochs 30
```

### Train with EfficientNet

```bash
python train.py \
    --model efficientnet_b0 \
    --data_dir ../data/processed \
    --save_dir ../models \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.0001
```

### Training Options

- `--model`: Model architecture (baseline, resnet18, resnet50, efficientnet_b0)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--device`: Device to use (cuda/cpu)

## Evaluation

### Evaluate a Trained Model

```bash
cd src
python evaluate.py \
    --model resnet50 \
    --checkpoint ../models/resnet50_best.pth \
    --data_dir ../data/processed \
    --results_dir ../results
```

This will generate:
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Classification report
- Saved results for further analysis

##  Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| Baseline CNN | TBD | TBD | TBD | TBD | ~2M |
| ResNet50 | TBD | TBD | TBD | TBD | ~23M |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | ~4M |

*Results will be updated after training*

## Implementation Details

### Data Augmentation
- Random horizontal/vertical flips
- Random rotation (±20°)
- Color jittering (brightness, contrast, saturation, hue)
- Random affine transformations
- Normalization using ImageNet statistics

### Model Architectures

**Baseline CNN:**
- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization after each conv layer
- Max pooling (2×2)
- 3 fully connected layers
- Dropout (p=0.5)

**Transfer Learning:**
- ResNet18/50/101
- EfficientNet B0-B4
- Fine-tuning last layers or full model
- Optional backbone freezing

### Hyperparameters
- Optimizer: Adam
- Learning rate: 1e-3 (baseline), 1e-4 (pretrained)
- Batch size: 32
- Image size: 224×224
- Loss function: Cross-entropy
- LR scheduler: ReduceLROnPlateau

## Notebooks

Explore the `notebooks/` directory for:
- Exploratory data analysis
- Model training experiments
- Results visualization
- Error analysis

## Authors

- Areeba Rashid
- Esha Sarfraz
- Vishakha Mishra

**Happy Classifying! 🍎🍌🍊**