"""Loss functions for neural networks."""

from typing import Optional
import genesis
from genesis import init
from genesis.tensor import Tensor
import genesis.nn.functional as F
from .module import Module


class SoftmaxLoss(Module):
    """
    Softmax loss.
    """
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass of the softmax loss.
        """
        num, classes = logits.shape
        mask = (y != -1)
        valid_logits = logits[mask] 
        valid_y = y[mask]

        y_one_hot = init.one_hot(classes, valid_y, dtype=logits.dtype, device=logits.device)
        logsum = F.logsumexp(valid_logits, axis=(1,))
        logits_y = F.summation(valid_logits * y_one_hot, axis=(1,))
        loss = logsum - logits_y
        return F.summation(loss) / valid_logits.shape[0]


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for classification tasks.
    
    Combines LogSoftmax and NLLLoss in a single class for numerical stability.
    
    Args:
        weight: Manual rescaling weight for each class
        ignore_index: Index to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        """Initialize CrossEntropyLoss.

        Args:
            weight: Manual rescaling weight for each class
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of cross-entropy loss.
        
        Args:
            input: Predicted logits of shape (N, C) where N is batch size, C is num classes
            target: Ground truth class indices of shape (N,)
            
        Returns:
            Scalar loss tensor
        """
        # Handle ignored indices
        if self.ignore_index != -100:
            mask = (target != self.ignore_index)
            input = input[mask]
            target = target[mask]
            if input.shape[0] == 0:
                return genesis.tensor(0.0, device=input.device, dtype=input.dtype)
        
        # Compute log-softmax for numerical stability
        log_prob = F.log_softmax(input, dim=1)
        
        # Create one-hot encoding for target
        num_classes = input.shape[1]
        target_one_hot = init.one_hot(num_classes, target, dtype=input.dtype, device=input.device)
        
        # Compute negative log-likelihood
        nll = -F.summation(log_prob * target_one_hot, axis=1)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Apply weight for each sample based on its class
            class_weights = self.weight[target]
            nll = nll * class_weights
        
        # Apply reduction
        if self.reduction == "none":
            return nll
        elif self.reduction == "sum":
            return F.summation(nll)
        elif self.reduction == "mean":
            return F.summation(nll) / nll.shape[0]
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MSELoss(Module):
    """
    Mean Squared Error loss for regression tasks.
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of MSE loss.
        
        Args:
            input: Predicted values
            target: Ground truth values
            
        Returns:
            Scalar loss tensor
        """
        # Compute squared error
        squared_error = (input - target) ** 2
        
        # Apply reduction
        if self.reduction == "none":
            return squared_error
        elif self.reduction == "sum":
            return F.summation(squared_error)
        elif self.reduction == "mean":
            return F.mean(squared_error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class L1Loss(Module):
    """
    L1 (Mean Absolute Error) loss for regression tasks.
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of L1 loss.
        
        Args:
            input: Predicted values
            target: Ground truth values
            
        Returns:
            Scalar loss tensor
        """
        # Compute absolute error
        abs_error = F.abs(input - target)
        
        # Apply reduction
        if self.reduction == "none":
            return abs_error
        elif self.reduction == "sum":
            return F.summation(abs_error)
        elif self.reduction == "mean":
            return F.mean(abs_error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class BCELoss(Module):
    """
    Binary Cross-Entropy loss for binary classification.
    
    Args:
        weight: Manual rescaling weight
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of BCE loss.
        
        Args:
            input: Predicted probabilities (should be in [0, 1])
            target: Ground truth binary labels (0 or 1)
            
        Returns:
            Scalar loss tensor
        """
        # Clamp input to avoid log(0) 
        eps = 1e-12
        input = F.clamp(input, eps, 1.0 - eps)
        
        # Compute binary cross entropy
        bce = -(target * F.log(input) + (1 - target) * F.log(1 - input))
        
        # Apply sample weights if provided
        if self.weight is not None:
            bce = bce * self.weight
        
        # Apply reduction
        if self.reduction == "none":
            return bce
        elif self.reduction == "sum":
            return F.summation(bce)
        elif self.reduction == "mean":
            return F.mean(bce)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class BCEWithLogitsLoss(Module):
    """
    Binary Cross-Entropy loss with logits for numerical stability.
    
    Combines sigmoid and BCE loss for better numerical stability.
    
    Args:
        weight: Manual rescaling weight
        reduction: Reduction method ('mean', 'sum', 'none')
        pos_weight: Weight for positive examples
    """
    
    def __init__(self, weight=None, reduction="mean", pos_weight=None):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of BCE with logits loss.
        
        Args:
            input: Predicted logits
            target: Ground truth binary labels (0 or 1)
            
        Returns:
            Scalar loss tensor
        """
        # Use log-sum-exp trick for numerical stability
        # BCE with logits: max(x, 0) - x * y + log(1 + exp(-abs(x)))
        max_val = F.maximum(input, 0)
        loss = max_val - input * target + F.log(1 + F.exp(-F.abs(input)))
        
        # Apply positive class weight if provided
        if self.pos_weight is not None:
            loss = target * self.pos_weight * loss + (1 - target) * loss
        
        # Apply sample weights if provided
        if self.weight is not None:
            loss = loss * self.weight
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return F.summation(loss)
        elif self.reduction == "mean":
            return F.mean(loss)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")