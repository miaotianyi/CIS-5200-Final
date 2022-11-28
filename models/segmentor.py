import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
import pytorch_lightning as pl

from .metrics import confusion_matrix_loss


class BaseSegmentor(pl.LightningModule):
    def __init__(self, net, learning_rate=1e-3):
        """
        Base segmentor for image segmentation tasks.

        This class contains methods for validation steps,
        training steps, optimizers, etc.

        Parameters
        ----------
        net : nn.Module
            Segmentation network architecture that maps
            inputs (tuple of tensors) to a [N, C, H, W] logit tensor output,
            where C is the number of labels.

            The spatial dimensions (H, W) can vary as you customize
            your network architecture.

            Note that the output tensor must be logit
            (real numbers from -infinity to infinity),
            not probabilities.

            In TGS Salt, this is ([N, 1, H, W], [N, 1]) -> [N, 1, H, W],
            since it's a binary segmentation.

        learning_rate : float
            Learning rate for Adam optimizer.
        """
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=['net']) # save hyperparms for wandblogger

    def forward(self, inputs):
        # this outputs probabilities
        # (not logits, which are only used in training)
        x = self.net(inputs)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, y = batch[:-1], batch[-1]
        y_pred = self.net(inputs)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, y = batch[:-1], batch[-1]
        y_pred_logit = self.net(inputs)
        self._log_validation_stats(y_true=y, y_pred_logit=y_pred_logit)

    def _log_validation_stats(self, y_true, y_pred_logit):
        y_pred_prob = torch.sigmoid(y_pred_logit)
        y_pred_label = (y_pred_prob > 0.5).float()
        self.log("val_bce", F.binary_cross_entropy_with_logits(input=y_pred_logit, target=y_true))
        self.log("val_focal", sigmoid_focal_loss(inputs=y_pred_logit, targets=y_true, reduction="mean"))
        self.log("val_accuracy", (y_pred_label == y_true).float().mean())
        self.log("val_sample_soft_f1_score",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="f_beta", average="samples"))
        self.log("val_sample_soft_dice_loss",
                 confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="dice", average="samples"))
        self.log("val_sample_soft_precision",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="precision", average="samples"))
        self.log("val_sample_soft_recall",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="recall", average="samples"))
        self.log("val_sample_hard_f1_score",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="f_beta", average="samples"))
        self.log("val_sample_hard_precision",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="precision", average="samples"))
        self.log("val_sample_hard_recall",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="recall", average="samples"))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
