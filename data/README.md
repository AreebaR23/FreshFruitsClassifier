# Data Directory

## Dataset Sources

### Kaggle Datasets
1. **Fruits Fresh and Rotten for Classification**
   - URL: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
   - Contains: Fresh and rotten fruit images

2. **Fruits Dataset**
   - URL: https://www.kaggle.com/datasets/moltean/fruits
   - Contains: Multiple fruit varieties

## Directory Structure
```
data/
├── raw/              # Original downloaded datasets
├── processed/        # Preprocessed and organized data
│   ├── train/
│   │   ├── fresh/
│   │   └── spoiled/
│   ├── val/
│   │   ├── fresh/
│   │   └── spoiled/
│   └── test/
│       ├── fresh/
│       └── spoiled/
└── README.md
```

## Setup Instructions

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Create new API token
   - Place `kaggle.json` in `~/.kaggle/`

3. Download datasets:
   ```bash
   kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
   kaggle datasets download -d moltean/fruits
   ```

4. Extract and organize:
   ```bash
   python ../src/prepare_data.py
   ```