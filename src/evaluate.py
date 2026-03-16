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
        for inputs, labels in tqdm(loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = 'binary')
    recall = recall_score(all_labels, all_preds, average = 'binary')
    f1 = f1_score(all_labels, all_preds, average = 'binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    print(f"\n{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*50}\n")
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return results


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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
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
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data for plotting
    data = {metric: [] for metric in metrics}
    for model_name in models:
        for metric in metrics:
            data[metric].append(results_dict[model_name][metric])
    
    # Create comparison plot
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    plt.show()
    
    # Print comparison table
    print("\nModel Comparison Table:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("=" * 70)
    for model_name in models:
        results = results_dict[model_name]
        print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1_score']:<12.4f}")

    

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
            inputs_device = inputs.to(device)
            outputs = model(inputs_device)
            _, preds = outputs.max(1)
            
            # Find misclassified
            mask = preds.cpu() != labels
            if mask.any():
                for i in range(len(mask)):
                    if mask[i] and len(misclassified) < num_samples:
                        misclassified.append({
                            'image': inputs[i],
                            'true_label': labels[i].item(),
                            'pred_label': preds[i].item(),
                            'confidence': torch.softmax(outputs[i], dim=0).max().item()
                        })
            
            if len(misclassified) >= num_samples:
                break
    
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
    class_names = class_names or ['fresh', 'spoiled']
    images = []
    preds = []
    labels = []
    num_classes = len(class_names)
    per_class_target = max(1, int(np.ceil(num_images / num_classes)))
    per_class_counts = {cls_idx: 0 for cls_idx in range(num_classes)}

    with torch.no_grad():
        for inputs, target in loader:
            inputs_device = inputs.to(device)
            outputs = model(inputs_device)
            _, predicted = outputs.max(1)

            for i in range(inputs.size(0)):
                if len(images) >= num_images:
                    break
                label = target[i].item()
                if per_class_counts.get(label, 0) >= per_class_target:
                    continue
                images.append(inputs[i])
                preds.append(predicted[i].item())
                labels.append(label)
                per_class_counts[label] = per_class_counts.get(label, 0) + 1

            if len(images) >= num_images:
                break

    # If we didn't collect enough (due to class imbalance), fill remaining slots
    if len(images) < num_images:
        with torch.no_grad():
            for inputs, target in loader:
                inputs_device = inputs.to(device)
                outputs = model(inputs_device)
                _, predicted = outputs.max(1)

                for i in range(inputs.size(0)):
                    if len(images) >= num_images:
                        break
                    images.append(inputs[i])
                    preds.append(predicted[i].item())
                    labels.append(target[i].item())

                if len(images) >= num_images:
                    break

    if not images:
        print("No images available to visualize.")
        return

    grid_size = int(np.ceil(np.sqrt(len(images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx >= len(images):
            continue
        img = _denormalize(images[idx])
        ax.imshow(img.permute(1, 2, 0))
        pred_label = class_names[preds[idx]]
        true_label = class_names[labels[idx]]
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction grid saved to {save_path}")
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
    parser = argparse.ArgumentParser(description='Evaluate FreshSense model')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--save_prediction_grid', action='store_true',
                       help='Save a grid image of predictions')
    parser.add_argument('--num_grid_images', type=int, default=16,
                       help='Number of images to include in the prediction grid')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Resolve paths relative to repo root when given as relative paths
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    checkpoint_path = Path(args.checkpoint)
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data only (no need for train/val splits during evaluation)
    _, val_transform = get_transforms(args.image_size, augment=False)
    test_dataset = FreshSenseDataset(
        os.path.join(str(data_dir), 'test'),
        transform=val_transform
    )
    if len(test_dataset) == 0:
        raise ValueError(
            "No test images found. Expected files under "
            f"{os.path.join(str(data_dir), 'test')}/[fresh|spoiled]. "
            "Run prepare_data.py to build the processed dataset or "
            "verify --data_dir points to the correct folder."
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(model_type=args.model, num_classes=2)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train a model first or pass the correct --checkpoint path."
        )
    model = load_checkpoint(model, str(checkpoint_path), device)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=['fresh', 'spoiled'],
        save_path=os.path.join(results_dir, f'{args.model}_confusion_matrix.png')
    )
    
    # Save results
    results_file = os.path.join(results_dir, f'{args.model}_results.pth')
    torch.save(results, results_file)
    print(f"\nResults saved to {results_file}")

    # Save prediction grid
    if args.save_prediction_grid:
        grid_path = os.path.join(results_dir, f'{args.model}_prediction_grid.png')
        save_prediction_grid(
            model,
            test_loader,
            device,
            class_names=['fresh', 'spoiled'],
            num_images=args.num_grid_images,
            save_path=grid_path
        )


if __name__ == '__main__':
    main()
