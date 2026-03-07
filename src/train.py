"""
Training script for FreshFruitsClassifier models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

from models import get_model, count_parameters
from data_loader import get_dataloaders
from utils import save_checkpoint, load_checkpoint, set_seed


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch.
    
    Steps for Implementation:
    1. Set model.train() to enable dropout/batchnorm updates.
    2. Initialize running loss and accuracy counters.
    3. Iterate over batches with a tqdm progress bar.
       - Move inputs/labels to device.
       - Zero gradients, forward pass, compute loss.
       - Backpropagate (loss.backward()) and optimizer.step().
       - Update running statistics and display in progress bar.
    4. Compute epoch-level loss/accuracy and return them.
    """

def validate(model, loader, criterion, device):
    """Validate the model.
    
    Steps for Implementation:
    1. Switch to eval mode and disable gradients with torch.no_grad().
    2. Iterate over validation batches and compute loss + predictions.
    3. Accumulate total loss and correct predictions.
    4. Return average loss and accuracy for the epoch.
    """
 


def train(
    model_type: str,
    data_dir: str,
    save_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    image_size: int = 224,
    augment: bool = True,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cuda',
    seed: int = 42,
    num_workers: int = 4,
    reuse_if_exists: bool = False
):
    """
    Main training function.
    
    Steps for Implementation:
    1. Seed all RNGs for reproducibility.
    2. Create the save directory for checkpoints/history.
    3. Resolve device (CUDA if available).
    4. Build dataloaders with augmentation for training.
    5. Instantiate the model via get_model and move to device.
    6. Configure loss, optimizer, and LR scheduler.
    7. For each epoch:
       - Run train_epoch and validate.
       - Step the scheduler based on val_loss.
       - Append history and print epoch summary.
       - Save the best checkpoint by validation accuracy.
    8. Save final checkpoint and training history.
    9. Return the trained model and history dict.
    
    Args:
        model_type: Type of model to train
        data_dir: Directory containing processed data
        save_dir: Directory to save models and results
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        image_size: Input image size
        augment: Whether to use data augmentation
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone layers (for transfer learning)
        device: Device to train on
        seed: Random seed
        num_workers: Number of data loading workers
        reuse_if_exists: Skip training when a best checkpoint already exists
    """


def main():
    """
    Entry point for CLI training.
    
    Steps for Implementation:
    1. Define CLI args for model, data paths, and hyperparameters.
    2. Parse args and map flags (no_augment, no_pretrained) to booleans.
    3. Call train() with parsed parameters.
    """


if __name__ == '__main__':
    main()
