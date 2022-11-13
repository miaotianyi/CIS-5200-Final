import torch
from torch import nn

from torch.nn.modules.utils import _pair


class MSConv2d(nn.Module):
    """
    Multi-source convolution block in 2D.

    Takes in 4 broadcastable tensors of shape
    ``[n,c1,h,w], [n,c2,h], [n,c3,w], [n,c4]`` respectively,
    outputs a tensor of shape ``[n, c_out, h_out, w_out]``.
    Some of these tensors may be omitted by setting ``c_i = 0``.

    Motivation: What if a CNN needs to take an image ``[n,c1,h,w]``
    and scalars ``[n,c4]`` as input at the same time?
    For example, each image has some useful metadata associated with it.

    Solution 1: Let the image go through multiple convolution layers
    in a traditional CNN; in the final fully-connected (linear) layer,
    concatenate the flattened CNN output and the scalar input.
    (See Google's Learning Hand-Eye Coordination for Robotic Grasping
    with Deep Learning and Large-Scale Data Collection)
    Drawback: The scalar information is not available in earlier layers.
    Also doesn't generalize for 1d ``[n,c2,h],[n,c3,w]`` metadata.

    Solution 2: Before feeding into the CNN, expand the scalars to shape
    ``[n,c4,h,w]`` and concatenate with image to become ``[n,c1+c4,h,w]``.
    The remaining architecture is the same as a vanilla CNN.
    This solution is very easy to implement.
    Drawback: Although ``torch.expand`` avoids copying the tensor in memory,
    the corresponding ``k*k-1`` filter parameters (where ``k`` is kernel size)
    are useless because there's only one constant.

    Solution 3: Use separate convolution blocks to create ``[n, c_out, h_out, w_out]`` and
    ``[n, c_out]`` tensors, which are broadcastable. Then add these output
    tensors as the final output of this module.
    This is mathematically equivalent to solution 3.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1,
                groups=1, bias=True, padding_mode="zeros",
                device=None, dtype=None):
        super(MSConv2d, self).__init__()
        # initialize hyperparameters
        self.in_channels = tuple(in_channels)
        if all(c <= 0 for c in self.in_channels):
            raise ValueError(f"All input channel sizes {self.in_channels} are non-positive.")
        self.nchw_channels, self.nch_channels, self.ncw_channels, self.nc_channels = in_channels
        self.out_channels = out_channels
        # same as Conv2D source code; make sure they are 2-tuples
        self.kernel_size = _pair(kernel_size)   # make sure kernel_size is 2-tuple
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        # other parameters
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        # device and dtype are propagated
        self.device = device
        self.dtype = dtype

        # initialize layers
        bias = self.bias    # since everything is added, only one output bias is needed
        if self.nchw_channels > 0:
            self.nchw_block = nn.Conv2d(
                in_channels=self.nchw_channels, out_channels=self.out_channels,
                kernel_size=self.kernel_size, stride=self.stride,
                padding=self.padding, dilation=self.dilation,
                groups=self.groups, bias=bias, padding_mode=self.padding_mode,
                device=self.device, dtype=self.dtype)
            bias = False
        else:
            self.nchw_block = None

        if self.nch_channels > 0:
            self.nch_block = nn.Conv1d(
                in_channels=self.nch_channels, out_channels=self.out_channels,
                kernel_size=self.kernel_size[0], stride=self.stride[0],
                padding=self.padding if isinstance(self.padding, str) else self.padding[0],
                dilation=self.dilation[0], groups=self.groups, bias=bias, padding_mode=self.padding_mode,
                device=self.device, dtype=self.dtype
            )
            bias = False
        else:
            self.nch_block = None

        if self.ncw_channels > 0:
            self.ncw_block = nn.Conv1d(
                in_channels=self.ncw_channels, out_channels=self.out_channels,
                kernel_size=self.kernel_size[1], stride=self.stride[1],
                padding=self.padding if isinstance(self.padding, str) else self.padding[1],
                dilation=self.dilation[1], groups=self.groups, bias=bias, padding_mode=self.padding_mode,
                device=self.device, dtype=self.dtype
            )
            bias = False
        else:
            self.ncw_block = None

        if self.nc_channels > 0:
            self.nc_block = nn.Linear(in_features=self.nc_channels, out_features=self.out_channels,
                                      bias=bias, device=self.device, dtype=self.dtype)
        else:
            self.nc_block = None

    def forward(self, *args):
        # tensor args are in order of (nchw, nch, ncw, nc), with possible omission
        i = 0
        y = None     # output tensor
        if self.nchw_block is not None:
            x = self.nchw_block(args[i])
            y = x if y is None else x + y
            i += 1
        if self.nch_block is not None:
            x = self.nch_block(args[i]).unsqueeze(3)    # nch -> nch1
            y = x if y is None else x + y
            i += 1
        if self.ncw_block is not None:
            x = self.ncw_block(args[i]).unsqueeze(2)    # ncw -> nc1w
            y = x if y is None else x + y
            i += 1
        if self.nc_block is not None:   # nc -> nc11
            x = self.nc_block(args[i]).unsqueeze(-1).unsqueeze(-1)
            y = x if y is None else x + y
        return y

