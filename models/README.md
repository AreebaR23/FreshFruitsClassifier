# Saved Models

This directory contains trained model checkpoints.

## File Naming Convention

Models are saved with the following naming pattern:
- `{model_name}_best.pth` - Best model based on validation accuracy
- `{model_name}_final.pth` - Final model after all epochs
- `{model_name}_history.pth` - Training history (loss and accuracy per epoch)

## Example Files

- `baseline_best.pth` - Best baseline CNN model
- `resnet50_best.pth` - Best ResNet-50 model
- `efficientnet_b0_best.pth` - Best EfficientNet-B0 model

## Loading Models

```python
from src.models import get_model
from src.utils import load_checkpoint
import torch

# Create model
model = get_model('resnet50', num_classes=2)

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_checkpoint(model, 'resnet50_best.pth', device)
```