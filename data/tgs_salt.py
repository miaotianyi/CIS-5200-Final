import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from matplotlib.image import imread


class SaltDataset(Dataset):
    """
    Training set in TGS Salt Dataset
    (https://www.kaggle.com/competitions/tgs-salt-identification-challenge)

    "train/images" contains 101 x 101 grayscale image input
    "train/masks" contains 101 x 101 binary segmentation labels
    "depths.csv" contains the
    """
    def __init__(self, normalize=True, dataset_dir="../datasets/TGS_Salt"):
        self.images_dir = os.path.join(dataset_dir, "train/images")
        self.masks_dir = os.path.join(dataset_dir, "train/masks")
        self.image_names = os.listdir(self.images_dir)  # the same for images and masks
        self.depths = pd.read_csv(os.path.join(dataset_dir, "depths.csv"))
        self.depths = self.depths.merge(
            pd.DataFrame({"id": [os.path.splitext(filename)[0] for filename in self.image_names]}),
            on="id",
            how="right"
        )
        self.min_depth = self.depths["z"].min()
        self.max_depth = self.depths["z"].max()

        if normalize:
            self.depths['z'] = self.depths['z'] / self.depths['z'].max()    # normalize depth
            self.depths['z'] = self.depths['z'].astype('float32')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        """
        Get an (image, depth, mask) tuple from dataset.

        Parameters
        ----------
        item : int
            Integer item index

        Returns
        -------
        x : np.ndarray
            Input image.
            Has shape [1, 101, 101]; has float32 value [0.0, 1.0]

        d : float
            Depth.

        y : np.ndarray
            Segmentation mask.
            Has shape [1, 101, 101]; has float32 value [0.0, 1.0]
        """
        filename = self.image_names[item]
        x = imread(os.path.join(self.images_dir, filename))     # h, w, c (grayscale but 3 channels)
        x = np.expand_dims(x[:, :, 0], 0)
        y = imread(os.path.join(self.masks_dir, filename))      # h, w (grayscale)
        y_mean = np.mean(y, axis=-1)                            # h
        y = np.expand_dims(y, 0)
        d = self.depths.loc[item, "z"]
        return x, d, y_mean, y



