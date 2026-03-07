"""
Evaluation utilities for FreshFruitsClassifier models.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

from models import get_model
from data_loader import FreshSenseDataset, get_transforms
from utils import load_checkpoint


def evaluate_model(model, loader, device, class_names=['fresh', 'spoiled']):
    """
    Evaluate model and compute metrics.
    
    Steps for Implementation:
    1. Switch to evaluation mode with model.eval() to disable dropout/batchnorm updates.
    2. Create containers for predictions, labels, and probabilities (lists work well).
    3. Iterate over the DataLoader with torch.no_grad() to avoid gradient tracking.
       - Move inputs to device (GPU/CPU).
       - Forward pass to get logits.
       - Convert logits to probabilities with softmax.
       - Compute predicted class via argmax.
       - Append predictions/labels/probabilities to accumulators.
    4. Convert lists to numpy arrays for sklearn compatibility.
    5. Compute metrics (accuracy/precision/recall/f1) and confusion matrix.
       - Use average='binary' for two classes; switch to 'macro' for multi-class.
    6. Print a readable summary + classification report.
    7. Return a results dictionary for downstream plotting/saving.
    
    Returns:
        Dictionary containing metrics and predictions
    """


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix.
    
    Steps for Implementation:
    1. Create a matplotlib figure with a reasonable size (e.g., 8x6).
    2. Use seaborn.heatmap to visualize counts with annotations.
    3. Label x/y axes with class names and add a title.
    4. Use tight_layout to prevent label clipping.
    5. If save_path is provided, save the figure before showing it.
    6. Call plt.show() to render interactively.
    """


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics.
    
    Steps for Implementation:
    1. Expect history to be a dict with keys: train_loss/val_loss/train_acc/val_acc.
    2. Create two subplots: one for loss, one for accuracy.
    3. Plot train/val curves with labels and grid for readability.
    4. Add axis labels and titles for each subplot.
    5. Use tight_layout to avoid overlap.
    6. Optionally save to disk, then display with plt.show().
    """
    

def compare_models(results_dict, save_path=None):
    """
    Compare multiple models.
    
    Steps for Implementation:
    1. Expect results_dict to map model_name -> metrics dict.
    2. Build a list of model names and target metrics (accuracy/precision/recall/f1).
    3. Create a bar chart with grouped bars for each metric.
       - Compute x positions and offsets per metric.
    4. Add labels, legend, and set y-limits to [0, 1].
    5. Save the plot if requested, then show it.
    6. Print a small metrics table to the console for quick inspection.
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save comparison plot
    """
    

def find_misclassified(model, loader, device, num_samples=10):
    """Find and return misclassified samples.
    
    Steps for Implementation:
    1. Set model to eval mode and iterate with torch.no_grad().
    2. Forward pass inputs to get logits and predicted class indices.
    3. Compare predictions with true labels to build a boolean mask.
    4. For each misclassified item (up to num_samples):
       - Save the image tensor (unnormalized if needed for display).
       - Save true label, predicted label, and confidence score.
    5. Stop early once num_samples have been collected.
    6. Return a list of dicts for downstream visualization.
    """
   

def _denormalize(image_tensor, mean=None, std=None):
    """
    Denormalize an image tensor for visualization.

    Args:
        image_tensor: Tensor of shape (C, H, W) normalized with mean/std
        mean: Channel-wise mean (defaults to ImageNet mean)
        std: Channel-wise std (defaults to ImageNet std)

    Returns:
        Tensor of shape (C, H, W) in [0, 1] range for plotting
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    return torch.clamp(image, 0, 1)


def save_prediction_grid(
    model,
    loader,
    device,
    class_names=None,
    num_images: int = 16,
    save_path: str = None
):
    """
    Save a grid image showing predictions (fresh vs spoiled).

    Steps for Implementation:
    1. Switch to eval mode and iterate batches with torch.no_grad().
    2. Collect up to num_images samples with their true/pred labels.
    3. Denormalize images for visualization.
    4. Plot a grid with matplotlib, titling each image with
       "Pred: <label> / True: <label>".
    5. Save to save_path if provided and show the figure.
    """


def main():
    """
    Entry point for CLI evaluation.
    
    Steps for Implementation:
    1. Define CLI arguments for model type, checkpoint, data paths, and hyperparams.
    2. Create the results directory if it doesn't exist.
    3. Resolve the device (prefer CUDA when available).
    4. Load test data with get_dataloaders(augment=False).
    5. Build the model architecture and load its checkpoint.
    6. Run evaluate_model to compute metrics and predictions.
    7. Plot and save the confusion matrix.
    8. Persist results to disk for later analysis.
    """
   

if __name__ == '__main__':
    main()
