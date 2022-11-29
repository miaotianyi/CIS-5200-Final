import torch
from torch import nn

class TrivialNet(nn.Module):
    def __init__(self):
        """
        A trivial segmentation CNN consisting of 2 layers.

        This is mostly used to see that there's no bug in training procedure

        Input is [N, 1, H, W] pixel in range [0.0, 1.0]
        Output is [N, 1, H, W] logits (unnormalized).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)

    def forward(self, inputs):
        x, d = inputs
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x