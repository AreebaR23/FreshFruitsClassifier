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
    """Baseline CNN model built from scratch."""
    
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
        super(BaselineCNN, self).__init__()
        
        # Use smaller filters for lightweight version
        filters = [16, 32, 64, 128] if lightweight else [32, 64, 128, 256]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_p)
        
        # Fully connected layers
        # Adaptive pooling keeps FC input size stable across image sizes
        fc_input = filters[3] * 4 * 4
        fc_hidden = 256 if lightweight else 512
        self.fc1 = nn.Linear(fc_input, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 64 if lightweight else 128)
        self.fc3 = nn.Linear(64 if lightweight else 128, num_classes)
    
    def forward(self, x):
        """
        Steps for Implementation:
        1. Apply each conv/bn/relu block followed by max pooling.
        2. Flatten the spatial feature map to (batch, features).
        3. Pass through FC layers with ReLU + dropout.
        4. Return logits (raw scores) for each class.
        """
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pool + flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

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

        super().__init__()
        model_fn = getattr(models, model_name)

        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None
        
        self.model = model_fn(weights=weights)

        if(freeze_backbone):
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    
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
    if model_type == 'baseline':
        return BaselineCNN(num_classes=num_classes, **kwargs)
    elif model_type.startswith('resnet'):
        return ResNetFineTune(
            num_classes=num_classes,
            model_name=model_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_type.startswith('efficientnet'):
        return EfficientNetFineTune(
            num_classes=num_classes,
            model_name=model_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Steps for Implementation:
    1. Iterate over model.parameters().
    2. Filter to parameters where requires_grad is True (trainable).
    3. Sum p.numel() for each trainable parameter tensor.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
