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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _,pred = torch.max(outputs, 1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': correct/total})

    epoch_loss = running_loss/total
    epoch_acc = correct/total

    return epoch_loss, epoch_acc






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
    # Set random seed
    set_seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint = os.path.join(save_dir, f'{model_type}_best.pth')
    if reuse_if_exists and os.path.exists(best_checkpoint):
        print(f"Found existing checkpoint at {best_checkpoint}. Skipping training.")
        model = get_model(
            model_type=model_type,
            num_classes=2,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = load_checkpoint(model, best_checkpoint, device)
        return model, None
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        image_size=image_size,
        augment=augment,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Val: {len(val_loader.dataset)} images")
    print(f"Test: {len(test_loader.dataset)} images")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = get_model(
        model_type=model_type,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(save_dir, f'{model_type}_best.pth')
            )
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Save final model and history
    save_checkpoint(
        model, optimizer, epochs, val_acc,
        os.path.join(save_dir, f'{model_type}_final.pth')
    )
    
    torch.save(history, os.path.join(save_dir, f'{model_type}_history.pth'))
    
    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")
    
    return model, history


def main():
    """
    Entry point for CLI training.
    
    Steps for Implementation:
    1. Define CLI args for model, data paths, and hyperparameters.
    2. Parse args and map flags (no_augment, no_pretrained) to booleans.
    3. Call train() with parsed parameters.
    """
    parser = argparse.ArgumentParser(description='Train FreshSense model')
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                       help='Data directory')
    parser.add_argument('--save_dir', type=str, default='../models',
                       help='Save directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (reduce for smaller machines)')
    parser.add_argument('--reuse_if_exists', action='store_true',
                       help='Skip training when a best checkpoint already exists')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight model for resource-constrained environments')
    
    args = parser.parse_args()

    # Resolve paths relative to repo root when given as relative paths
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    if not data_dir.is_absolute():
        if '..' in data_dir.parts:
            data_dir = (Path.cwd() / data_dir).resolve()
        else:
            data_dir = (repo_root / data_dir).resolve()
    if not save_dir.is_absolute():
        if '..' in save_dir.parts:
            save_dir = (Path.cwd() / save_dir).resolve()
        else:
            save_dir = (repo_root / save_dir).resolve()
    
    train(
        model_type=args.model,
        data_dir=str(data_dir),
        save_dir=str(save_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        augment=not args.no_augment,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        reuse_if_exists=args.reuse_if_exists
    )

if __name__ == '__main__':
    main()
