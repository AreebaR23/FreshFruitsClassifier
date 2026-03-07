"""
Utility functions.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Steps for implementation:
    1. Seed Python's random module and NumPy RNG.
    2. Seed PyTorch CPU RNG with torch.manual_seed.
    3. Seed all CUDA RNGs with torch.cuda.manual_seed_all.
    4. Set cudnn.deterministic=True and cudnn.benchmark=False
       to reduce nondeterminism across runs.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    




def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint.
    
    Steps for implementation:
    1. Create a dict with epoch, model_state_dict (model.state_dict() for torchvision models), optimizer_state_dict, val_acc.
    2. Serialize to disk with torch.save at the given path.
    """
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc}
    torch.save(checkpoint, path)


def load_checkpoint(model, path, device, optimizer=None):
    """Load model checkpoint.
    
    Steps for implementation:
    1. Load the checkpoint with torch.load(map_location=device).
    2. Restore model weights with load_state_dict.
    3. If optimizer provided, restore optimizer state.
    4. Move model to device and set to eval mode.
    5. Optionally print the saved epoch and val accuracy for traceability.
    6. Return the updated model.
    """



def get_lr(optimizer):
    """Get current learning rate from optimizer.
    
    Steps for implementation:
    1. Iterate over optimizer.param_groups (usually one group).
    2. Return the 'lr' value from the first param group.
    """
    

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        Steps for implementation:
        1. Reset val, avg, sum, and count to zero.
        2. Call this at the start of each epoch.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Steps for implementation:
        1. Set current value to val.
        2. Accumulate sum += val * n and count += n.
        3. Recompute avg as sum / count.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count