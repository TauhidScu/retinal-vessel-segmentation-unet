import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss implementation with boundary clamping.
    """
    def __init__(self, eps=1e-6):
        super(BCELoss, self).__init__()
        self.eps = eps

    def forward(self, pr, gt):
        # Validate inputs
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth labels must be 0 or 1."
        assert pr.shape == gt.shape, "Predictions and ground truth must have the same shape."
        
        # Clamp predictions to prevent numerical instability
        pr = torch.clamp(pr, self.eps, 1 - self.eps)
        bce_loss = F.binary_cross_entropy(pr, gt, reduction='mean')
        return bce_loss


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation tasks.
    Measures overlap between predicted and ground truth masks.
    """
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Smoothing factor to avoid division by zero

    def forward(self, pr, gt):
        # Validate inputs
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth must be 0 or 1."
        assert pr.shape == gt.shape, "Predictions and ground truth must have the same shape."
        
        # Flatten tensors for calculation
        pr_flat = torch.flatten(pr)
        gt_flat = torch.flatten(gt)
        
        # Calculate Dice coefficient
        intersection = torch.sum(pr_flat * gt_flat)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(pr_flat) + torch.sum(gt_flat) + self.smooth)
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss using weighted sum of BCE and Dice losses.
    Useful for segmentation tasks to balance pixel-wise and region-based accuracy.
    """
    def __init__(self, bce_loss_weight=0.25, dice_loss_weight=0.75):
        super(CombinedLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight

    def forward(self, pr, gt):
        # Validate inputs
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth labels must be 0 or 1"
        assert pr.shape == gt.shape, "Prediction and ground truth must have the same shape"

        # Calculate individual losses
        bce = self.bce_loss(pr, gt)
        dice = self.dice_loss(pr, gt)

        # Combine losses with weights
        combined_loss = self.bce_loss_weight * bce + self.dice_loss_weight * dice
        return combined_loss
