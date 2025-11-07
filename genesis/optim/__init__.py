"""Optimization algorithms and learning rate scheduling for Genesis.

This package provides PyTorch-compatible optimizers and learning rate schedulers
for training neural networks with automatic differentiation.
"""

from .optimizer import SGD, Adam
from .adamw import AdamW
from . import lr_scheduler
