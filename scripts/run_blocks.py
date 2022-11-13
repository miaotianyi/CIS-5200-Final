"""
Test whether the blocks run properly
"""
import numpy as np
from itertools import product

import torch

from models.blocks import MSConv2d


base_in_channels = (2, 3, 4, 5)
for mask in product((0, 1), repeat=4):
    mask = np.array(mask)
    if mask.sum() == 0:
        continue  # at least one positive
    in_channels = (mask * base_in_channels).tolist()
    block = MSConv2d(in_channels, 6, kernel_size=3)
    # prepare input x
    xs = []
    if mask[0]:  # nchw
        xs.append(torch.rand(10, base_in_channels[0], 7, 8))
    if mask[1]:  # nchw
        xs.append(torch.rand(10, base_in_channels[1], 7))
    if mask[2]:  # nchw
        xs.append(torch.rand(10, base_in_channels[2], 8))
    if mask[3]:
        xs.append(torch.rand(10, base_in_channels[3]))
    y = block(*xs)
    print(tuple(mask), tuple(y.shape))
