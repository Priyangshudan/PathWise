"""Compact U-Net for binary footpath-damage segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolution, batch-normalization and ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Four-level U-Net returning one-channel raw logits.

    ``base_channels=16`` gives feature widths 16, 32, 64, 128 and 256. The
    decoder aligns odd-sized feature maps to their skip tensors when needed.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if out_channels != 1:
            raise ValueError("This baseline is for binary segmentation and requires out_channels=1")

        widths = [base_channels * (2**i) for i in range(5)]
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(widths[3], widths[4])

        self.up4 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(widths[4], widths[3])
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(widths[3], widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(widths[2], widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(widths[1], widths[0])
        self.output = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    @staticmethod
    def _align(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat((self._align(self.up4(b), e4), e4), dim=1))
        d3 = self.dec3(torch.cat((self._align(self.up3(d4), e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._align(self.up2(d3), e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._align(self.up1(d2), e1), e1), dim=1))
        return self.output(d1)
