import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.u1 = DoubleConv(256+128, 128)
        self.u2 = DoubleConv(128+64, 64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))

        u1 = self.up(c3)
        u1 = self.u1(torch.cat([u1, c2], dim=1))

        u2 = self.up(u1)
        u2 = self.u2(torch.cat([u2, c1], dim=1))

        return self.out(u2)
