"""
Data loading and preprocessing utilities:

This module implements the data pipeline for loading and preprocessing food images:
1. FreshFruitsClassifier: Custom PyTorch Dataset class for loading images from directory structure
2. get_transforms(): Creates data augmentation pipelines for training/validation
3. get_dataloaders(): Wraps datasets in DataLoader objects for batching and parallel loading
4. compute_mean_std(): Calculates dataset statistics for normalization

Key Concepts:
- PyTorch Dataset: Defines how to load individual samples (images + labels)
- PyTorch DataLoader: Handles batching, shuffling, and parallel loading
- Transforms: Image preprocessing and augmentation operations
- Normalization: Scaling pixel values using mean/std for better training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import numpy as np


class FreshSenseDataset(Dataset):
    """
    Custom PyTorch Dataset for fresh/spoiled food classification.
    
    Steps for implementation:
    1. Inherit from torch.utils.data.Dataset
    2. Implement __init__: Initialize paths, transforms, and scan directory structure
    3. Implement __len__: Return total number of samples
    4. Implement __getitem__: Load and return a single image-label pair
    
    Directory Structure Expected:
        data_dir/
            fresh/
                image1.jpg
                image2.jpg
            spoiled/
                image3.jpg
                image4.jpg
    """
    
    def __init__(self, root_dir: str, transform=None, class_names=None):
        """
        Initialize the dataset by scanning the directory structure.
        
        Steps for implementation:
        1. Store root directory path and transforms
        2. Define class names (categories): ['fresh', 'spoiled']
        3. Create class_to_idx mapping: {'fresh': 0, 'spoiled': 1}
        4. Initialize empty list to store (image_path, label) tuples
        5. Call helper method to scan directories and populate samples list
        
        Args:
            root_dir: Directory with subdirectories for each class
            transform: Optional transforms to apply (data augmentation, normalization)
            class_names: List of class names (default: ['fresh', 'spoiled'])
        
        Example:
            dataset = FreshSenseDataset('data/train', transform=my_transforms)
        """
        
        #Store directory + Transforms
        self.root_dir = root_dir
        self.transform = transform
        
        # Default class names
        self.class_names = class_names or ['fresh', 'spoiled']
        
        # Map class names to numeric labels
        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        
        # List of img path / label
        self.samples = []
        
        # Load img path + label
        self._load_samples()
        
    
    def _load_samples(self):
        """
        Scan directory structure and load all image paths with their labels.
        
        Steps for implementation:
        1. Loop through each class name (e.g., 'fresh', 'spoiled')
        2. Construct full path to class directory (e.g., 'data/train/fresh')
        3. Check if directory exists (skip if missing)
        4. Get numeric label for this class from class_to_idx mapping
        5. List all files in the class directory
        6. Filter for image files (.jpg, .png, .jpeg extensions)
        7. For each image, store tuple of (full_image_path, numeric_label)
        
        Result: self.samples = [
            ('data/train/fresh/img1.jpg', 0),
            ('data/train/fresh/img2.jpg', 0),
            ('data/train/spoiled/img3.jpg', 1),
            ...
        ]
        """
        
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Steps for implementation:
        1. Return length of self.samples list
        
        Required by PyTorch DataLoader to determine batch scheduling.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return a single sample (image, label) at the given index.
        
        Steps for implementation:
        1. Get the (image_path, label) tuple from self.samples[idx]
        2. Load image from disk using PIL (Python Imaging Library)
        3. Convert to RGB format (ensures 3 channels, handles grayscale/RGBA)
        4. Apply transforms if provided (resize, augment, normalize, convert to tensor)
        5. Return (image_tensor, label) tuple
        
        Args:
            idx: Index of sample to retrieve (0 to len-1)
        
        Returns:
            image: Transformed image tensor of shape (C, H, W)
            label: Integer class label (0 or 1)
        
        Called by PyTorch DataLoader during training/validation batching.
        """


def get_transforms(image_size: int = 224, augment: bool = True):
    """
    Create transformation pipelines for training and validation data.
    
    Steps for implementation:
    1. Create training transform pipeline:
       - Resize to fixed size (e.g., 224x224)
       - Apply data augmentation (if enabled): flips, rotations, color jitter
       - Convert PIL Image to PyTorch tensor (HWC -> CHW, [0-255] -> [0-1])
       - Normalize using ImageNet mean/std
    
    2. Create validation transform pipeline:
       - Resize to fixed size (no augmentation)
       - Convert to tensor
       - Normalize (same as training)
    
    3. Return both pipelines
    
    Data Augmentation Benefits:
    - Reduces overfitting by creating variations of training images
    - Improves model generalization to unseen data
    - Simulates real-world variations (lighting, orientation, etc.)
    
    Normalization:
    - Uses ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Standardizes inputs for pretrained models
    - Speeds up convergence during training
    
    Args:
        image_size: Target image size (square: width=height)
        augment: Whether to apply data augmentation (True for training)
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    #Normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Training transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness = 0.2,
                contrast = 0.2,
                saturation = 0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
    # Validation + test transforms (NO augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    return train_transform, val_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader objects for train, validation, and test sets.
    
    Steps for implementation:
    1. Get transform pipelines (with/without augmentation)
    2. Create 3 Dataset objects (train, val, test) with appropriate transforms
    3. Wrap each Dataset in a DataLoader with proper settings:
       - Train: shuffle=True (randomize order each epoch)
       - Val/Test: shuffle=False (consistent order for reproducibility)
       - Set batch_size, num_workers for parallel loading
    4. Return all three DataLoaders
    
    DataLoader Benefits:
    - Automatic batching: Groups samples into batches for efficient GPU processing
    - Shuffling: Randomizes training order to reduce overfitting
    - Parallel loading: Uses multiple CPU workers to load data while GPU trains
    - Pin memory: Speeds up CPU-to-GPU transfer (use when GPU available)
    
    Directory Structure Expected:
        data_dir/
            train/
                fresh/
                spoiled/
            val/
                fresh/
                spoiled/
            test/
                fresh/
                spoiled/
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Number of samples per batch (higher = faster but more memory)
        image_size: Target image size for resizing
        augment: Whether to apply data augmentation to training set
        num_workers: Number of parallel workers for data loading (0=main thread only)
        pin_memory: Whether to use pinned memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """


def compute_mean_std(data_dir: str, image_size: int = 224):
    """
    Compute per-channel mean and standard deviation of the dataset.
    
    Steps for implementation:
    1. Create basic transforms (resize + to_tensor, NO normalization)
    2. Create dataset and dataloader
    3. Initialize accumulators for mean and std (3 channels: R, G, B)
    4. Loop through all batches:
       - Reshape batch: (B, C, H, W) -> (B, C, H*W)
       - Compute mean and std across spatial dimensions
       - Accumulate statistics
    5. Divide by total number of images to get average
    6. Return mean and std as numpy arrays
    
    WHY THIS IS USEFUL:
    - Custom normalization: Use your dataset's statistics instead of ImageNet
    - Better for domain-specific data (e.g., medical images, satellite imagery)
    - Formula: normalized = (pixel - mean) / std
    
    HOW NORMALIZATION WORKS:
    - Centers data around 0 (subtract mean)
    - Scales variance to 1 (divide by std)
    - Helps neural network training converge faster
    - Each channel (R, G, B) normalized independently
    
    Args:
        data_dir: Directory containing the images
        image_size: Size to resize images to
    
    Returns:
        Tuple of (mean, std) as numpy arrays, shape (3,) for RGB channels
        Example: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    """
    