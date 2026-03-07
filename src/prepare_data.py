"""
Data preparation script for organizing Kaggle datasets into train/val/test splits.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple
import argparse


def create_directory_structure(base_dir: str):
    """Create train/val/test directory structure.
    
    Steps for implementation:
    1. Define split names (train/val/test) and class names (fresh/spoiled).
    2. Loop over each split and class to build the target path.
    3. Use os.makedirs(..., exist_ok=True) to create all directories safely.
    4. Print a summary so the user knows where outputs were created.
    """
    splits = ['train', 'test', 'val']
    classes = ['fresh', 'spoiled']
    p = os.path.join(base_dir, "processed")
    if(os.path.isdir(p) == False):
        os.mkdir(p)

    for s in splits:
        for c in classes:
            path = os.path.join(p, s, c)
            os.makedirs(path, exist_ok=True)
    
    print('Output directory located in ', p)
    
    


def split_dataset(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_class: int = None
):
    """
    Split dataset into train/val/test sets.
    
    Steps for implementation:
    1. Set the random seed for reproducible splits.
    2. For each class directory:
       - List image files with allowed extensions.
       - Optionally cap to max_samples_per_class for smaller machines.
       - Shuffle the list before splitting.
    3. Compute counts for train/val/test based on ratios.
    4. Copy files into processed/{split}/{class} folders using shutil.copy2.
    5. Print a per-class summary of split sizes.
    
    Args:
        source_dir: Directory containing fresh/spoiled subdirectories
        dest_dir: Destination directory for processed data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    pass
   
def organize_kaggle_dataset(raw_dir: str, processed_dir: str):
    """
    Organize Kaggle fruit dataset into fresh/spoiled categories.
    
    This function should be customized based on the actual structure
    of your downloaded Kaggle datasets.
    
     Steps for implementation:
     1. Walk the raw_dir and classify directories as fresh or spoiled
         based on folder names (fresh/good vs rotten/bad/spoiled).
     2. Create a temporary directory with fresh/ and spoiled/ subfolders.
     3. Copy image files from the discovered folders into temp_dir.
         - Rename files to avoid name collisions.
     4. Return temp_dir so it can be split into train/val/test.
    """
    pass

def main():
    """
    Entry point for dataset preparation.
    
    Steps for implementation:
    1. Parse CLI args for raw/processed paths and split ratios.
    2. Create the processed folder structure.
    3. Organize raw Kaggle data into fresh/spoiled temp folders.
    4. Split the temp folder into train/val/test.
    5. Clean up the temp directory to save space.
    6. Print completion message.
    """

    create_directory_structure(r'C:\Users\binom\OneDrive\Desktop\FreshFruits\FreshFruitsClassifier\data')


if __name__ == '__main__':
    main()