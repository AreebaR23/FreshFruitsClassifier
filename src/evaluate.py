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
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = 'binary')
    recall = recall_score(all_labels, all_preds, average = 'binary')
    f1 = f1_score(all_labels, all_preds, average = 'binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n Evaluation Results")
    print("-------------------")
    print(f"Accuracy: {acc: .4f}")
    print(f"Precision: {precision: .4f}")
    print(f"Recall: {recall: .4f}")
    print(f"F1 Score: {f1: .4f} \n")
    print(classification_report(all_labels, all_preds, target_names = class_names))
    
    return{
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": np.array(all_probs)
    }


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
    plt.figure(figsize = (8, 6))
    
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        xticklabels = class_names,
        yticklabels = class_names
    )
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


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
    plt.figure(figsize = (10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label = "Train Loss")
    plt.plot(history["val_loss"], label = "Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label = "Train Acc")
    plt.plot(history["val_acc"], label = "Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

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
    
    models = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = {m: [results_dict[model] [m] for model in models] for m in metrics}
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.figure(figsize = (10, 6))
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[metric], width, label = metric)
        
    plt.xticks(x + width * 1.5, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    print("\n Model Comparison:")
    for model in models:
        print(model, results_dict[model])
    

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
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            
            mask = preds != labels
            
            for i in range(len(inputs)):
                if mask[i]:
                    misclassified.append({
                        "image": inputs[i].cpu(),
                        "true": labels[i].item(),
                        "pred": preds[i].item(),
                        "confidence": probs[i] [preds[i]].item()
                    })
                    if len(misclassified) >= num_samples:
                        return misclassified
                    
    return misclassified
   

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
    model.eval()
    
    images = []
    preds = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim = 1)
            predictions = torch.argmax(probs, dim = 1)
            
            for i in range(len(inputs)):
                images.append(_denormalize(inputs[i]))
                preds.append(predictions[i].item())
                labels.append(targets[i].item())
                
                if len(images) >= num_images:
                    break
            
            if len(images) >= num_images:
                break
    
    cols = 4
    rows = int(np.ceil(num_images / cols))
    
    plt.figure(figsize = (12, 8))
    
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")
        
        pred_label = class_names[preds[i]]
        true_label = class_names[labels[i]]
        
        plt.title(f"Pred: {pred_label}\n True: {true_label}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    


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
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = True)
    parser.add_argument("--batch_size", type = int, default = 32)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(train = False)
    
    dataset = FreshSenseDataset(args.data_dir, transform = transforms)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False
    )
    
    model = get_model(args.model)
    model = load_checkpoint(model, args.checkpoint, device)
    
    results = evaluate_model(model, loader, device)
    results_dir = Path("Confusion Results")
    results_dir.mkdir(exist_ok = True)
    
    plot_confusion_matrix(
        results["confusion_matrix"],
        ["fresh", "spoiled"],
        save_path = results_dir / "confusion_matrix.png"
    )


if __name__ == '__main__':
    main()
