import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Optional, List

# Constants for segmentation modes
BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"


def soft_tversky_score(output, target, alpha=0.5, beta=0.5, smooth=1e-6, eps=1e-7, dims=[2, 3]):
    """
    Compute the Tversky score between output and target tensors.
    
    Args:
        output: Model output tensor
        target: Ground truth tensor
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothness constant
        eps: Small epsilon for numerical stability
        dims: Dimensions to sum over (default: [2, 3] for height and width in 4D tensors)
    """
    # Ensure output and target are of the same shape
    assert output.shape == target.shape, "Output and target tensors must have the same shape"
    
    # Compute intersection and sum along the specified dimensions
    intersection = (output * target).sum(dim=dims)
    
    # Compute false positive and false negative
    false_positive = (output * (1 - target)).sum(dim=dims)
    false_negative = ((1 - output) * target).sum(dim=dims)
    
    # Compute Tversky score
    tversky_index = (intersection + smooth) / (intersection + alpha * false_positive + beta * false_negative + smooth)
    
    return tversky_index


def soft_dice_score(output, target, smooth=1e-6, eps=1e-7, dims=[2, 3]):
    """
    Compute the Dice score (F1) using Tversky with alpha=beta=0.5
    """
    return soft_tversky_score(output, target, 0.5, 0.5, smooth, eps, dims)


def to_tensor(x, dtype=None) -> torch.Tensor:
    """
    Convert input to a torch tensor.
    
    Args:
        x: Input data (tensor, numpy array, list, or tuple)
        dtype: Optional desired dtype for the tensor
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


class DiceLoss(_Loss):
    """
    Dice loss for image segmentation tasks.
    Supports binary, multiclass, and multilabel cases.
    """
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes: List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `-log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for computing the Dice loss.
        
        Args:
            y_pred: Predicted tensor of shape (N, C, H, W)
            y_true: Target tensor of shape (N, H, W) or (N, C, H, W)
            
        Returns:
            Computed loss value
        """
        assert y_true.size(0) == y_pred.size(0)

        # Apply activations to get [0..1] class probabilities if needed
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        # Process inputs based on segmentation mode
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        elif self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)

        elif self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        # Compute Dice score
        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        # Convert score to loss
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Zero contribution of channels that don't have true pixels
        # (Dice loss is undefined for empty classes)
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        # Filter by specified classes if needed
        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        """Aggregate the loss by taking the mean"""
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        """Compute the Dice score between output and target"""
        return soft_dice_score(output, target, smooth, eps, dims)
