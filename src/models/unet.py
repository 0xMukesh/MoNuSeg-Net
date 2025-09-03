import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, reduction_rate: int = 16
    ) -> None:
        super().__init__()

        self.reduction_rate = reduction_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels // self.reduction_rate,
                kernel_size=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels // self.reduction_rate,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),
            nn.Sigmoid(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        if in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.identity_mapping = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_mapping(x)

        x = self.conv1(x)
        x = self.conv2(x)
        se = self.se(x)
        x = x * se

        x = x + identity
        x = self.relu(x)

        return x


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int) -> None:
        super().__init__()

        self.wg = nn.Sequential(
            nn.Conv2d(
                in_channels=g_channels, out_channels=x_channels, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(num_features=x_channels),
        )

        self.wx = nn.Sequential(
            nn.Conv2d(
                in_channels=x_channels, out_channels=x_channels, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(num_features=x_channels),
        )

        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx, wx = x.shape[2], x.shape[3]
        hg, wg = g.shape[2], g.shape[3]

        if hx != hg or wx != wg:
            g = F.interpolate(g, size=(hx, wx), mode="bilinear", align_corners=False)

        g = self.wg(g)
        x1 = self.wx(x)
        out = self.relu(g + x1)
        out = self.phi(out)
        out = out * x

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip = self.double_conv(x)
        x = self.pool(skip)

        return [x, skip]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()

        self.upscale = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )

        self.attn_gate = AttentionGate(
            g_channels=out_channels, x_channels=skip_channels
        )

        self.double_conv = DoubleConv(
            in_channels=out_channels * 2, out_channels=out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x)
        skip = self.attn_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: List[int] = [64, 128, 256, 512, 1024],
    ) -> None:
        super().__init__()

        self.features = features
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bottleneck = DoubleConv(
            in_channels=self.features[-1], out_channels=self.features[-1] * 2
        )

        for feature in self.features:
            self.encoders.append(
                EncoderBlock(in_channels=in_channels, out_channels=feature)
            )

            in_channels = feature

        for feature in reversed(self.features):
            self.decoders.append(
                DecoderBlock(
                    in_channels=feature * 2, out_channels=feature, skip_channels=feature
                )
            )

        self.final_conv = nn.Conv2d(
            in_channels=self.features[0], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        x = self.final_conv(x)

        return x
