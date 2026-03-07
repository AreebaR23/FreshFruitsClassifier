"""
Model architectures for FreshFruitsClassifier: baseline CNN and pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import ssl
import urllib.request

# Handle SSL certificate issues for model downloads
ssl._create_default_https_context = ssl._create_unverified_context
from typing import Optional


class BaselineCNN(nn.Module):
    """Building Baseline CNN model from scratch."""
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5, lightweight: bool = False):
        """
        Args:
            num_classes: Number of output classes (default: 2 for fresh/spoiled)
            dropout_p: Dropout probability
            lightweight: Use smaller model for resource-constrained environments
        
        Steps for Implementation:
        1. Call super().__init__() to initialize nn.Module internals.
        2. Choose filter sizes based on lightweight flag.
        3. Create sequential convolutional blocks:
           - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
        4. Add Dropout for regularization before fully connected layers.
        5. Compute the flattened feature size after pooling to size FC layers.
        6. Create FC layers that end with num_classes outputs.
        """
        
    
    def forward(self, x):
        """
        Steps for Implementation:
        1. Apply each conv/bn/relu block followed by max pooling.
        2. Flatten the spatial feature map to (batch, features).
        3. Pass through FC layers with ReLU + dropout.
        4. Return logits (raw scores) for each class.
        """


class ResNetFineTune(nn.Module):
    """ResNet model with fine-tuning capability."""
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', etc.)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers
        
        Steps for Implementation:
        1. Choose the correct torchvision ResNet constructor based on model_name.
        2. Pass weights when pretrained=True (None otherwise).
        3. Optionally freeze backbone parameters by disabling requires_grad.
        4. Replace the final fully connected layer with a new Linear layer
           that outputs num_classes logits.
        """
       
    
    def forward(self, x):
        """Forward pass through the underlying torchvision model."""
        return self.model(x)


class EfficientNetFineTune(nn.Module):
    """EfficientNet model with fine-tuning capability."""
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers
        
        Steps for Implementation:
        1. Select the proper torchvision EfficientNet constructor.
        2. Load ImageNet weights when pretrained=True.
        3. Optionally freeze all backbone parameters (feature extractor).
        4. Replace classifier[1] with a Linear layer to match num_classes.
        """
    
    def forward(self, x):
        """Forward pass through the underlying torchvision model."""
        return self.model(x)


def get_model(
    model_type: str = 'baseline',
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Steps for Implementation:
    1. Inspect model_type to choose a matching architecture.
    2. Instantiate BaselineCNN for 'baseline' and pass kwargs (e.g., lightweight).
    3. Instantiate ResNetFineTune for resnet* types with pretrained/freeze options.
    4. Instantiate EfficientNetFineTune for efficientnet* types.
    5. Raise ValueError for unsupported model types to fail fast.
    
    Args:
        model_type: Type of model ('baseline', 'resnet18', 'resnet50', 'efficientnet_b0', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for transfer learning models)
        freeze_backbone: Whether to freeze backbone layers
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        PyTorch model
    """


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Steps for Implementation:
    1. Iterate over model.parameters().
    2. Filter to parameters where requires_grad is True (trainable).
    3. Sum p.numel() for each trainable parameter tensor.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
