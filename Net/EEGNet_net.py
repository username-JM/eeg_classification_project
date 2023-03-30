
import torch.nn as nn
from layers import LinearWithConstraint, Conv2dWithConstraint, ConvSamePad2d

class EEGNet(nn.Module):
    def __init__(self, args, shape):
        super(EEGNet, self).__init__()
        self.num_ch = shape[2]
        self.F1 = 16
        self.F2 = 32
        self.D = 2
        self.sr = 250
        self.P1 = 4
        self.P2 = 8
        self.t1 = 16
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size=(1, self.sr//2), bias=False, padding='same'),
            nn.BatchNorm2d(self.F1)
        )
        # Spatial conv (Depth-wise conv, EEGNet 2nd block)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.num_ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P1))
        )
        # Separable conv (EEGNet 3rd block)
        self.separable_conv = nn.Sequential(
            # depth-wise
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, self.t1), groups=self.F1 * self.D, bias=False),
            # point-wise
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P2))
        )
        # Dense
        self.linear = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(in_features=self.F2 * 33, out_features=4, max_norm=0.25)
        )


    def forward(self, x):
        out = self.temporal_conv(x)
        out = self.spatial_conv(out)
        out = self.separable_conv(out)
        out = self.linear(out)
        return out