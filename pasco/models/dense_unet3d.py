"""
Dense 3D UNet for semantic segmentation.
Pure PyTorch implementation without MinkowskiEngine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock3D(nn.Module):
    """Residual block with two 3D convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class EncoderBlock(nn.Module):
    """Encoder block: downsample + residual blocks."""

    def __init__(self, in_ch, out_ch, n_res_blocks=2):
        super().__init__()
        self.down = ConvBlock3D(in_ch, out_ch, kernel_size=2, stride=2, padding=0)
        self.res_blocks = nn.Sequential(*[ResBlock3D(out_ch) for _ in range(n_res_blocks)])

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + residual blocks."""

    def __init__(self, in_ch, skip_ch, out_ch, n_res_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # After concat with skip connection
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)
        self.res_blocks = nn.Sequential(*[ResBlock3D(out_ch) for _ in range(n_res_blocks)])

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res_blocks(x)
        return x


class DenseUNet3D(nn.Module):
    """
    Dense 3D UNet for semantic segmentation.

    Architecture:
        Input: [B, 1, H, W, D] occupancy grid
        Encoder: 4 levels with 2x downsampling each
        Bottleneck: Multi-kernel convolutions
        Decoder: 4 levels with 2x upsampling + skip connections
        Output: [B, n_classes, H, W, D] logits

    Args:
        in_channels: Number of input channels (1 for occupancy)
        n_classes: Number of output classes
        base_channels: Base channel count (doubled at each encoder level)
        n_res_blocks: Number of residual blocks per level
    """

    def __init__(self, in_channels=1, n_classes=71, base_channels=32, n_res_blocks=2):
        super().__init__()

        self.n_classes = n_classes
        c = base_channels  # 32

        # Initial convolution
        self.init_conv = nn.Sequential(
            ConvBlock3D(in_channels, c),
            ResBlock3D(c),
        )

        # Encoder: [c, 2c, 4c, 8c]
        self.enc1 = EncoderBlock(c, c * 2, n_res_blocks)      # /2
        self.enc2 = EncoderBlock(c * 2, c * 4, n_res_blocks)  # /4
        self.enc3 = EncoderBlock(c * 4, c * 8, n_res_blocks)  # /8
        self.enc4 = EncoderBlock(c * 8, c * 16, n_res_blocks) # /16

        # Bottleneck with multi-kernel convolutions
        self.bottleneck = nn.Sequential(
            ConvBlock3D(c * 16, c * 16, kernel_size=3, padding=1),
            ConvBlock3D(c * 16, c * 16, kernel_size=3, padding=1),
        )

        # Decoder
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8, n_res_blocks)   # /8
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4, n_res_blocks)    # /4
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2, n_res_blocks)    # /2
        self.dec1 = DecoderBlock(c * 2, c, c, n_res_blocks)            # /1

        # Output head
        self.out_conv = nn.Conv3d(c, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W, D] input occupancy grid

        Returns:
            logits: [B, n_classes, H, W, D] class logits
        """
        # Initial
        e0 = self.init_conv(x)  # [B, c, H, W, D]

        # Encoder
        e1 = self.enc1(e0)  # [B, 2c, H/2, W/2, D/2]
        e2 = self.enc2(e1)  # [B, 4c, H/4, W/4, D/4]
        e3 = self.enc3(e2)  # [B, 8c, H/8, W/8, D/8]
        e4 = self.enc4(e3)  # [B, 16c, H/16, W/16, D/16]

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b, e3)   # [B, 8c, H/8, W/8, D/8]
        d3 = self.dec3(d4, e2)  # [B, 4c, H/4, W/4, D/4]
        d2 = self.dec2(d3, e1)  # [B, 2c, H/2, W/2, D/2]
        d1 = self.dec1(d2, e0)  # [B, c, H, W, D]

        # Output
        logits = self.out_conv(d1)  # [B, n_classes, H, W, D]

        return logits

    def get_loss(self, logits, labels, class_weights=None, ignore_index=255):
        """
        Compute cross-entropy loss.

        Args:
            logits: [B, n_classes, H, W, D]
            labels: [B, H, W, D] with class indices
            class_weights: Optional tensor of class weights
            ignore_index: Label to ignore in loss computation

        Returns:
            loss: Scalar loss value
        """
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        return criterion(logits, labels)


class DenseUNet3DLight(nn.Module):
    """
    Lighter version of Dense 3D UNet with fewer levels.
    Better for memory-constrained scenarios.

    Architecture:
        Input: [B, 1, H, W, D]
        3 encoder levels (8x total downsampling)
        3 decoder levels
        Output: [B, n_classes, H, W, D]
    """

    def __init__(self, in_channels=1, n_classes=71, base_channels=32, n_res_blocks=2):
        super().__init__()

        self.n_classes = n_classes
        c = base_channels

        # Initial
        self.init_conv = nn.Sequential(
            ConvBlock3D(in_channels, c),
            ResBlock3D(c),
        )

        # Encoder: 3 levels [c, 2c, 4c]
        self.enc1 = EncoderBlock(c, c * 2, n_res_blocks)      # /2
        self.enc2 = EncoderBlock(c * 2, c * 4, n_res_blocks)  # /4
        self.enc3 = EncoderBlock(c * 4, c * 8, n_res_blocks)  # /8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock3D(c * 8, c * 8, kernel_size=3, padding=1),
            ConvBlock3D(c * 8, c * 8, kernel_size=3, padding=1),
        )

        # Decoder
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4, n_res_blocks)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2, n_res_blocks)
        self.dec1 = DecoderBlock(c * 2, c, c, n_res_blocks)

        # Output
        self.out_conv = nn.Conv3d(c, n_classes, kernel_size=1)

    def forward(self, x):
        # Initial
        e0 = self.init_conv(x)

        # Encoder
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d3 = self.dec3(b, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Output
        return self.out_conv(d1)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = DenseUNet3D(in_channels=1, n_classes=71, base_channels=32).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with small input
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
