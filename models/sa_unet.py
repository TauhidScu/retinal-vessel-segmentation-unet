import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import drop_block2d


class ChannelPool(nn.Module):
    """Pool channel information by combining max and mean features"""
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    """Spatial attention mechanism that helps model focus on relevant areas"""
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with optional attention mechanism"""
    def __init__(self, in_channels, out_channels, mid_channels=None, attention=False):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.bnrelu1 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.attention = SpatialGate() if attention else nn.Identity()

        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bnrelu2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = drop_block2d(x, 0.18, 7, inplace=True, training=self.training)
        x = self.bnrelu1(x)
        x = self.attention(x)
        x = self.conv2(x)
        x = drop_block2d(x, 0.18, 7, inplace=True, training=self.training)
        x = self.bnrelu2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, attention=attention),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=3, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size differences for skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate skip connection with upsampled feature map
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution layer to produce output"""
    def __init__(self, in_channels, out_channels=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SAUNet(nn.Module):
    """U-Net architecture with Spatial Attention mechanism"""
    def __init__(self):
        super(SAUNet, self).__init__()
        # Encoder path
        self.inc = DoubleConv(3, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128, attention=True)
        
        # Decoder path with skip connections
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        
        # Output projection
        self.outc = OutConv(16)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        # Encoder pathway
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder pathway with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output processing
        logits = self.outc(x)
        output = self.activate(logits)
        return output
