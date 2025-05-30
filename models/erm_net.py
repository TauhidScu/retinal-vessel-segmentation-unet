import torch
import torch.nn as nn
import torchvision.models as models
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

class ResNet18Encoder(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9, sync_channels=False, weights=models.ResNet18_Weights.DEFAULT):
        super(ResNet18Encoder, self).__init__()
        resnet = models.resnet18(weights=weights)

        # Modify the first convolution to accept larger inputs
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Encoder uses ResNet-18 blocks with DropBlock2D
        self.initial = nn.Sequential(
            resnet.conv1,  # First convolution
            resnet.bn1,    # BatchNorm
            resnet.relu,   # ReLU
            resnet.maxpool # MaxPool
        )
        self.dropblock = DropBlock2D(block_size, keep_prob, sync_channels)

        # Use ResNet-18 encoding blocks
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

    def forward(self, x):
        x_initial = self.initial(x)
        x_initial = self.dropblock(x_initial)  # Apply DropBlock after the initial conv

        x1 = self.encoder1(x_initial)
        x1 = self.dropblock(x1)  # Apply DropBlock after the first block

        x2 = self.encoder2(x1)
        x2 = self.dropblock(x2)  # Apply DropBlock after the second block

        x3 = self.encoder3(x2)
        x3 = self.dropblock(x3)  # Apply DropBlock after the third block

        return x_initial, x1, x2, x3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, sync_channels=False):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False)
        #self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        # Conv->Drop->BN
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropblock1 = DropBlock2D(block_size, keep_prob)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        # Conv(1*1)->Conv->Drop->BN
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropblock2 = DropBlock2D(block_size, keep_prob)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        # ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('Shape before deconv: ', x.shape)
        x = self.deconv(x)
        #print('Shape after deconv: ', x.shape)
        x= self.conv1(x)
        x =self.dropblock1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2_1(x)
        x = self.conv2(x)
        x = self.dropblock2(x)  # Apply DropBlock in the decoder block
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x

class UNetResNet18(nn.Module):
    def __init__(self, n_classes=1, block_size=7, keep_prob=0.9, sync_channels=False):
        super(UNetResNet18, self).__init__()
        self.encoder = ResNet18Encoder(block_size, keep_prob, sync_channels)
        
        # Bottleneck between encoder and decoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            DropBlock2D(block_size, keep_prob, sync_channels)  # Apply DropBlock in the bottleneck
        )
        
        # Decoder blocks
        self.decoder3 = DecoderBlock(512, 256, block_size, keep_prob, sync_channels)
        self.decoder2 = DecoderBlock(256 + 128, 128, block_size, keep_prob, sync_channels)
        self.decoder1 = DecoderBlock(128+64, 64, block_size, keep_prob, sync_channels)
        
        # Final upsampling and output layer
        self.final_conv = nn.Conv2d(64 + 64, n_classes, kernel_size=1)
        self.active = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Encoding path
        #print('shape of input: ', x.shape)
        x_initial, x1, x2, x3 = self.encoder(x)
        #print('shape of x_initial, x1, x2 and x3: ', x_initial.shape, x1.shape, x2.shape, x3.shape)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x3)
        
        # Decoding path with skip connections
        x = self.decoder3(x_bottleneck)
        x = torch.cat([x, x2], dim=1)  # Skip connection from encoder2
        
        x = self.decoder2(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection from encoder1
        
        x = self.decoder1(x)
        
        # Ensure the sizes of x and x_initial match before concatenating
        x_initial = F.interpolate(x_initial, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_initial], dim=1)  # Skip connection from initial
        
        # Final 1x1 convolution to produce output
        x_out = self.final_conv(x)
        #print('x_out shape: ', x_out.shape)
        x_out = self.active(x_out)
        return x_out

# # Test with random input tensor
# model = UNetResNet18(n_classes=1, block_size=7, keep_prob=0.9, sync_channels=False)
# x = torch.randn(1, 3, 1024, 1536)  # Example input tensor with batch_size=1, channels=3, height=256, width=256
# output = model(x)
# print(output.shape)  # Should give shape: (1, n_classes, 256, 256)
