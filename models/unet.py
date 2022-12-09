import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP

# adapted from https://github.com/milesial/Pytorch-UNet

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, feature_dim=64, bilinear=False, meta_dim=0, meta_layer=None, d_dim=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_dim = feature_dim
        self.meta_dim = meta_dim
        self.meta_layer = meta_layer
        self.d_dim = d_dim

        if meta_layer == 'bneck' and meta_dim != d_dim:
            # hardcoding the bottleneck dimension
            d_enc_dim = 6
            self.meta_encoder = MLP(d_dim, [max(d_dim, d_enc_dim), max(d_dim, d_enc_dim)], d_enc_dim)
        elif meta_layer is not None:
            self.meta_encoder = MLP(d_dim, [d_dim, d_dim], d_dim)
        
        in_channels = n_channels + meta_dim if meta_layer == 'begin' else n_channels
        self.inc = DoubleConv(in_channels, self.feature_dim)
        self.down1 = Down(self.feature_dim, self.feature_dim*2)
        self.down2 = Down(self.feature_dim*2, self.feature_dim*4)
        self.down3 = Down(self.feature_dim*4, self.feature_dim*8)
        factor = 2 if bilinear else 1
        self.down4  = DownFuse(self.feature_dim*8, (self.feature_dim*16) // factor, meta_dim=meta_dim) \
            if meta_layer == 'bneck' else Down(self.feature_dim*8, (self.feature_dim*16) // factor)
        self.up1 = Up(self.feature_dim*16, (self.feature_dim*8) // factor, bilinear)
        self.up2 = Up(self.feature_dim*8, (self.feature_dim*4) // factor, bilinear)
        self.up3 = Up(self.feature_dim*4, (self.feature_dim*2) // factor, bilinear)
        self.up4 = Up(self.feature_dim*2, self.feature_dim, bilinear)
        out_channels = self.feature_dim + meta_dim if meta_layer == 'end' else self.feature_dim
        self.outc = OutConv(out_channels, n_classes)

    def fuse_inputs(self, x, d):
        bneck_dim = (x.shape[-2], x.shape[-1])
        # assume d is averaged over W ! 
        if self.meta_dim == d.shape[-1]: 
            tiled_d = d[:, :, None, None].repeat(1, 1, bneck_dim[0], bneck_dim[1]) # NH -> NH11 -> NHHW
        elif self.meta_dim == 1 and d.shape[-1] == 101:
            tiled_d = d[:, :, None].repeat(1, 1, bneck_dim[1])[:, None, :, :] # NH -> NHW -> N1HW
        else:
            NotImplementedError
        fused_x = torch.cat([x, tiled_d], dim=1) #concatenate along channel dimension
        return fused_x

    def forward(self, inputs):
        x, d = inputs
        if self.meta_layer == 'begin':
            d = self.meta_encoder(d)
            x = self.fuse_inputs(x, d)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.meta_layer == 'bneck':
            d = self.meta_encoder(d)
            x5 = self.down4(x4, d)
        else:
            x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.meta_layer == 'end':
            d = self.meta_encoder(d)
            x = self.fuse_inputs(x, d)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel=3, pad=1):
        super().__init__()

        self.convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convrelu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    # def _forward_meta(self, x, d):
    #     x = self.convrelu1(x)
    #     bneck_dim = (x.shape[-2], x.shape[-1])
    #     tiled_d = d[:, :, None, None].repeat(1, 1, bneck_dim[0], bneck_dim[1])
    #     fused_x = torch.cat([x, tiled_d], dim=1) #concatenate along channel dimension
    #     x = self.convrelu2(fused_x)
    #     return x

    def forward(self, x):
        x = self.convrelu1(x)
        x = self.convrelu2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel=3, pad=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel=kernel, pad=pad)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class DownFuse(nn.Module):
    """Downscaling with maxpool then double conv with fusion with robot encoding"""

    def __init__(self, in_channels, out_channels, kernel=3, pad=1, meta_dim=0):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.meta_dim = meta_dim

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fuseconvrelu = nn.Sequential(
            nn.Conv2d(out_channels+meta_dim, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, d):
        x = self.downsample(x) # BxCxHxW
        x = self.convrelu(x)
        # print(x.shape)
        bneck_dim = (x.shape[-2], x.shape[-1])

        if self.meta_dim == 0: # if no fuse at all
            fuse_input = x
        else: # if pose or ft is fused
            # assume d is averaged over W ! 
            if self.meta_dim == d.shape[-1]: 
                tiled_d = d[:, :, None, None].repeat(1, 1, bneck_dim[0], bneck_dim[1]) # NH -> NH11 -> NHHW
            elif self.meta_dim == 1 and d.shape[-1] == bneck_dim[0]:
                tiled_d = d[:, :, None].repeat(1, 1, bneck_dim[1])[:, None, :, :] # NH -> NHW -> N1HW

            else:
                NotImplementedError
            # tiled_encoding = d.reshape(-1, self.meta_dim, 1, 1).repeat(1, 1, bneck_dim[0], bneck_dim[1])
            fuse_input = torch.cat([x, tiled_d], dim=1) #concatenate along channel dimension

        x = self.fuseconvrelu(fuse_input) 
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
