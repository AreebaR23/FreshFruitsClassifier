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
    random.seed(seed)
    
    # Define classes
    classes = ['fresh', 'spoiled']
    allowed_ext = ('.jpg', '.jpeg', '.png')
    
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        
        # List images w/allowed extensions
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(allowed_ext)
        ]
        
        # Limit dataset size if needed
        if max_samples_per_class:
            images = images[:max_samples_per_class]
        
        # Shuffle images before splitting to randomize
        random.shuffle(images)
        
        total = len(images)
        
        # Compute number of images for each split
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        # Slice lists to create each split
        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        # Create dictionary for easier iteration
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        # Copy files to respective processed dirs
        for split_name, file_list in splits.items():
            for file in file_list:
                src = os.path.join(class_dir, file)
                dst = os.path.join(dest_dir, split_name, cls, file)
                
                shutil.copy2(src, dst)
            
        # Print summary for this cls/iteration
        print(f"{cls}: train = {len(train_files)}, val = {len(val_files)}, test = {len(test_files)}")
   
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
    
    # Create temp folders to hold all images sorted by class
    temp_dir = os.path.join(processed_dir, "temp")
    fresh_dir = os.path.join(temp_dir, "fresh")
    spoiled_dir = os.path.join(temp_dir, "spoiled")
    
    os.makedirs(fresh_dir, exist_ok = True)
    os.makedirs(spoiled_dir, exist_ok = True)
    
    allowed_ext = ('.jpg', '.jpeg', '.png')
    counter = 0
    
    # Iterate through raw dataset folder
    for root, dirs, files in os.walk(raw_dir):
        folder_name = os.path.basename(root).lower()
        
        # Determine if folder is fresh/spoiled
        if "fresh" in folder_name or "good" in folder_name:
            target = fresh_dir
        elif "rotten" in folder_name or "spoiled" in folder_name or "bad" in folder_name:
            target = spoiled_dir
        else:
            continue # skip folders that don't match either
        
        # Copy images into temp folder (rename to avoid collisions)
        for file in files:
            if file.lower().endswith(allowed_ext):
                src = os.path.join(root, file)
                new_name = f"{counter}_{file}"
                dst = os.path.join(target, new_name)
                
                shutil.copy2(src, dst)
                counter += 1
    
    # Debugging
    # print("Temporary dataset created at:", temp_dir)
    
    # Return temp directory path for splitting 
    return temp_dir

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
    parser = argparse.ArgumentParser(description='Prepare FreshSense dataset')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='Directory containing raw Kaggle datasets')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per class (for smaller machines)')
    
    args = parser.parse_args()

    # Resolve paths relative to repo root when given as relative paths
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    if not raw_dir.is_absolute():
        if '..' in raw_dir.parts:
            raw_dir = (Path.cwd() / raw_dir).resolve()
        else:
            raw_dir = (repo_root / raw_dir).resolve()
    if not processed_dir.is_absolute():
        if '..' in processed_dir.parts:
            processed_dir = (Path.cwd() / processed_dir).resolve()
        else:
            processed_dir = (repo_root / processed_dir).resolve()
    
    # Creating directory structure
    create_directory_structure(str(processed_dir.parent))
    
    # Organizing Kaggle dataset
    print("Organizing Kaggle dataset...")
    temp_dir = organize_kaggle_dataset(str(raw_dir), str(processed_dir))
    
    # Splitting into train/val/test
    print("\nSplitting dataset...")
    split_dataset(
        temp_dir,
        str(processed_dir),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.max_samples
    )
    
    # Cleaning up temporary directory
    shutil.rmtree(temp_dir)
    
    print("\nDataset preparation complete!")

    # create_directory_structure(r'C:\Users\binom\OneDrive\Desktop\FreshFruits\FreshFruitsClassifier\data')

if __name__ == '__main__':
    main()