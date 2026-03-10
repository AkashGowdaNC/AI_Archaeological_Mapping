import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=4):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)

        up = nn.functional.interpolate(d2, scale_factor=2)

        up = self.up1(up)

        output = self.final(up)

        return output
