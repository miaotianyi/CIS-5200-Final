import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from models.segmentor import BaseSegmentor
from data.tgs_salt import SaltDataset


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


def main():
    # get dataset
    dataset = SaltDataset()
    # set random seed
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [0.7, 0.3], generator=seed)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    trainer = pl.Trainer(
        max_epochs=6,
        devices=1, accelerator="gpu"
    )
    trainer.fit(model=BaseSegmentor(TrivialNet()),
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    # use this command to see tensorboard:
    # tensorboard --logdir=scripts/lightning_logs/
    main()
