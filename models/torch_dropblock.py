import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """
    Implementation of DropBlock2D regularization technique.
    
    DropBlock is a form of structured dropout where entire regions of feature maps
    are dropped together instead of individual pixels.
    """
    
    def __init__(self, block_size, keep_prob, sync_channels=False):
        """
        Initialize DropBlock2D.
        
        Args:
            block_size (int): Size of the blocks to drop.
            keep_prob (float): Probability of keeping a block.
            sync_channels (bool): Whether to use the same mask across all channels.
        """
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels

    def forward(self, x):
        # Don't apply dropout during evaluation or if keep_prob is 1
        if not self.training or self.keep_prob == 1:
            return x
        else:
            # Compute drop probability adjusted for block size
            gamma = self._compute_gamma(x)
            
            # Create random mask based on gamma
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)
            mask = mask.unsqueeze(1)

            # Apply mask across channels based on sync_channels setting
            if self.sync_channels:
                # Use the same mask for all channels
                mask = mask.repeat(1, x.shape[1], 1, 1)
            else:
                # Use channel-specific masks
                mask = mask.expand_as(x)

            # Use max pooling to extend the mask to block_size
            mask = F.max_pool2d(mask, kernel_size=self.block_size, 
                               stride=1, padding=self.block_size // 2)
            # Invert the mask (1 for keep, 0 for drop)
            mask = 1 - mask

            # Apply the mask and rescale to maintain mean feature magnitude
            x = x * mask * (mask.numel() / mask.sum())
            
            return x

    def _compute_gamma(self, x):
        """
        Compute the gamma value (drop probability per pixel)
        adjusted for feature map and block sizes.
        """
        # Base drop probability per pixel
        gamma = (1.0 - self.keep_prob) / (self.block_size ** 2)
        
        # Scale by feature map size ratio to maintain overall drop rate
        gamma *= x.shape[2] * x.shape[3] / ((x.shape[2] - self.block_size + 1) * 
                                           (x.shape[3] - self.block_size + 1))
        return gamma
