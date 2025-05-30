import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, block_size, keep_prob, sync_channels=False):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)
            mask = mask.unsqueeze(1)

            if self.sync_channels:
                mask = mask.repeat(1, x.shape[1], 1, 1)
            else:
                mask = mask.expand_as(x)

            mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
            mask = 1 - mask

            x = x * mask * (mask.numel() / mask.sum())  # Re-scaling
            return x

    def _compute_gamma(self, x):
        gamma = (1.0 - self.keep_prob) / (self.block_size ** 2)
        gamma *= x.shape[2] * x.shape[3] / ((x.shape[2] - self.block_size + 1) * (x.shape[3] - self.block_size + 1))
        return gamma
