import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(BCELoss, self).__init__()
        self.eps = eps

    def forward(self, pr, gt):
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth labels must be 0 or 1."
        assert pr.shape == gt.shape, "Predictions and ground truth must have the same shape."
        pr = torch.clamp(pr, self.eps, 1 - self.eps)
        bce_loss = F.binary_cross_entropy(pr, gt, reduction='mean')
        return bce_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pr, gt):
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth must be 0 or 1."
        assert pr.shape == gt.shape
        pr_flat = torch.flatten(pr)
        gt_flat = torch.flatten(gt)
        intersection = torch.sum(pr_flat * gt_flat)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(pr_flat) + torch.sum(gt_flat) + self.smooth)
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, bce_loss_weight=0.25, dice_loss_weight=0.75):
        super(CombinedLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight

    def forward(self, pr, gt):
        assert torch.all((gt == 0.0) | (gt == 1.0)), "Ground truth labels must be 0 or 1"
        assert pr.shape == gt.shape, "Prediction and ground truth must have the same shape"

        bce = self.bce_loss(pr, gt)
        dice = self.dice_loss(pr, gt)

        combined_loss = self.bce_loss_weight * bce + self.dice_loss_weight * dice
        return combined_loss


# # Define dummy predictions and ground truth tensors (Batch Size = 1, Channels = 1, Height = 256, Width = 256)
# pr_valid = torch.sigmoid(torch.randn(1, 1, 256, 256))  # Simulated prediction (output of a sigmoid)
# gt_valid = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Binary ground truth (0 or 1)

# pr_invalid_shape = torch.sigmoid(torch.randn(1, 1, 256, 255))  # Mismatched shape
# gt_invalid_values = torch.randint(-1, 3, (1, 1, 256, 256)).float()  # Invalid values (-1, 0, 1, 2)

# # Instantiate loss functions
# bce_loss = BCELoss()
# dice_loss = DiceLoss()
# combined_loss = CombinedLoss()

# # Function to test loss functions
# def test_loss_functions(pr, gt, description):
#     try:
#         print(f"\nTesting with {description}...")
#         bce = bce_loss(pr, gt)
#         print(f"BCELoss Value: {bce.item()}")
        
#         dice = dice_loss(pr, gt)
#         print(f"DiceLoss Value: {dice.item()}")
        
#         combined = combined_loss(pr, gt)
#         print(f"CombinedLoss Value: {combined.item()}")
#     except AssertionError as e:
#         print(f"AssertionError: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# # Test with valid data
# test_loss_functions(pr_valid, gt_valid, "valid data")

# # Test with mismatched shapes
# test_loss_functions(pr_invalid_shape, gt_valid, "mismatched shapes")

# # Test with invalid ground truth values
# test_loss_functions(pr_valid, gt_invalid_values, "invalid ground truth values")