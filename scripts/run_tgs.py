import os

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.ops import sigmoid_focal_loss

import pytorch_lightning as pl

from models.metrics import confusion_matrix_loss
from dataloaders.tgs_salt import SaltDataset


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

    def forward(self, x, d):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class SaltSegmentor(pl.LightningModule):
    def __init__(self, net):
        """

        Parameters
        ----------
        net : nn.Module
            Segmentation architecture that maps x, d -> y (y is logit)
            [N, 1, H, W], [N, 1] -> [N, 1, H, W]
        """
        super().__init__()
        self.net = net

    def forward(self, x, d):
        # this outputs probabilities
        # (not logits, which are only used in training)
        x = self.net(x, d)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, d, y = batch
        y_pred = self.net(x, d)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, d, y = batch
        y_pred_logit = self.net(x, d)
        y_pred_prob = F.sigmoid(y_pred_logit)
        bce = F.binary_cross_entropy_with_logits(input=y_pred_logit, target=y)
        focal_loss = sigmoid_focal_loss(inputs=y_pred_logit, targets=y, reduction="mean")
        f1_loss = confusion_matrix_loss(y_true=y, y_pred=y_pred_prob, metric="f_beta", average="samples")
        dice_loss = confusion_matrix_loss(y_true=y, y_pred=y_pred_prob, metric="dice", average="samples")
        self.log("val_bce", bce)
        self.log("val_focal", focal_loss)
        self.log("val_f1", 1 - f1_loss)
        self.log("val_dice_loss", dice_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# get dataset
dataset = SaltDataset()
# set random seed
seed = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset, [0.7, 0.3], generator=seed)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(train_set, batch_size=32, shuffle=True)
trainer = pl.Trainer(
    max_epochs=10,
    devices=1, accelerator="gpu"
)
trainer.fit(model=SaltSegmentor(TrivialNet()),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
