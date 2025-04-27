import torch
import torch.nn as nn

class RotEquivariantConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(RotEquivariantConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out0 = self.conv(x)
        x90 = torch.rot90(x, k=1, dims=[2, 3])
        out90 = self.conv(x90)
        out90 = torch.rot90(out90, k=3, dims=[2, 3])
        x180 = torch.rot90(x, k=2, dims=[2, 3])
        out180 = self.conv(x180)
        out180 = torch.rot90(out180, k=2, dims=[2, 3])
        x270 = torch.rot90(x, k=3, dims=[2, 3])
        out270 = self.conv(x270)
        out270 = torch.rot90(out270, k=1, dims=[2, 3])
        out = (out0 + out90 + out180 + out270) / 4
        return out

class GCNN(nn.Module):
    def __init__(self):
        super(GCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            RotEquivariantConv(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            RotEquivariantConv(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            RotEquivariantConv(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x