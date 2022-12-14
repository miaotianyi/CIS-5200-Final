import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
import pytorch_lightning as pl

import wandb

from .metrics import confusion_matrix_loss

from models.trivial import TrivialNet
from models.unet import UNet
from models.ResNet import ResNet, BasicBlock, Bottleneck
from models.embedding import SinusoidalPositionEmbeddings

class BaseSegmentor(pl.LightningModule):
    def __init__(self, model, learning_rate, meta_dim, pos_embed, embed_dim, use_ymean, d_dim, threshold=0.5,**kwargs):
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

        # if use position embedding
        self.pos_embed = None
        if pos_embed:
            self.pos_embed = SinusoidalPositionEmbeddings(embed_dim)
            meta_dim = embed_dim

        # if use y mean as meta data
        self.use_ymean = use_ymean

        if model == 'trivial':
            self.model = TrivialNet()
        elif model == 'unet':
            self.model = UNet()
        elif model == 'unet_begin':
            self.model = UNet(meta_layer='begin', meta_dim=meta_dim, d_dim=d_dim)
        elif model == 'unet_bneck':
            self.model = UNet(meta_layer='bneck', meta_dim=meta_dim, d_dim=d_dim)
        elif model == 'unet_end':
            self.model = UNet(meta_layer='end', meta_dim=meta_dim, d_dim=d_dim)
        elif model == 'resnet':
            self.model = ResNet(1, BasicBlock, [1, 1, 1, 1], num_classes=1)
        elif model == 'resnet_34':
            self.model = ResNet(1, BasicBlock, [3, 4, 6, 3], num_classes=1)
        elif model == 'resnet_begin':
            self.model = ResNet(1, BasicBlock, [1, 1, 1, 1], num_classes=1, meta_layer='begin', meta_dim=meta_dim)
        elif model == 'resnet_end':
            self.model = ResNet(1, BasicBlock, [1, 1, 1, 1], num_classes=1, meta_layer='end', meta_dim=meta_dim)
        else:
            raise ValueError('Unknown model: {}'.format(model))

        self.learning_rate = learning_rate
        self.threshold = threshold
        self.save_hyperparameters() # save hyperparms for wandblogger

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BaseSegmentor')
        parser.add_argument("--model", type=str, default='trivial')
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--normalize_depth", type=bool, default=False)
        parser.add_argument("--d_dim", type=int, default=0)
        parser.add_argument("--meta_dim", type=int, default=0)

        return parent_parser
    
    def forward(self, inputs):
        # this outputs probabilities
        # (not logits, which are only used in training)

        # position embedding
        if self.pos_embed is not None:
            x, d, _ = inputs
            inputs = (x, self.pos_embed(d))

        if self.use_ymean:
            x, _, y_mean = inputs
            inputs = (x, y_mean)

        x = self.model(inputs)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, y = batch[:-1], batch[-1]

        # position embedding
        if self.pos_embed is not None:
            x, d, _ = inputs
            inputs = (x, self.pos_embed(d))

        if self.use_ymean:
            x, _, y_mean = inputs
            inputs = (x, y_mean)
        else:
            x, d, _ = inputs
            inputs = (x, d)

        y_pred_logit = self.model(inputs)
        loss = F.binary_cross_entropy_with_logits(input=y_pred_logit, target=y)
        self.log("train_loss", loss)
        self._log_training_stats(y_true=y, y_pred_logit=y_pred_logit)
        # self._log_images(input_im = inputs[0], y_true=y, y_pred_logit=y_pred_logit, prefix="train_")
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, y = batch[:-1], batch[-1]

        # position embedding
        if self.pos_embed is not None:
            x, d, _ = inputs
            inputs = (x, self.pos_embed(d))

        if self.use_ymean:
            x, _, y_mean = inputs
            inputs = (x, y_mean)
            
        y_pred_logit = self.model(inputs)
        self._log_validation_stats(y_true=y, y_pred_logit=y_pred_logit)
        # self._log_images(input_im = inputs[0], y_true=y, y_pred_logit=y_pred_logit, prefix="val_")
    
    def _log_images(self, input_im, y_true, y_pred_logit, prefix):
        """
        Log images to wandb.

        Parameters
        ----------
        input_im : torch.Tensor
            Input image tensor of shape [N, C, H, W].
        y_true : torch.Tensor
            True label tensor of shape [N, H, W].
        y_pred_logit : torch.Tensor
            Predicted label tensor of shape [N, H, W].
        """
        y_pred = torch.sigmoid(y_pred_logit)
        y_pred_label = (y_pred > self.threshold).float()

        # hack to convert smoothed labels to binary
        y_true_label = (y_true > self.threshold).float()
        
        self.logger.experiment.log({
            prefix + "input_im": wandb.Image(input_im[0]),
            prefix + "y_true": wandb.Image(y_true[0]),
            prefix + "y_pred": wandb.Image(y_pred[0]),
            prefix + "y_pred_label": wandb.Image(y_pred_label[0]),
            prefix + "y_true_label": wandb.Image(y_true_label[0]),
        })

    def _log_training_stats(self, y_true, y_pred_logit):
        y_pred_prob = torch.sigmoid(y_pred_logit)
        y_pred_label = (y_pred_prob > self.threshold).float()

        # hack to convert smoothed labels to binary
        y_true = (y_true > self.threshold).float()

        self.log("train_bce", F.binary_cross_entropy_with_logits(input=y_pred_logit, target=y_true), on_step=False , on_epoch=True)
        self.log("train_focal", sigmoid_focal_loss(inputs=y_pred_logit, targets=y_true, reduction="mean"), on_step=False, on_epoch=True)
        self.log("train_accuracy", (y_pred_label == y_true).float().mean(), on_step=False, on_epoch=True)
        self.log("train_sample_soft_f1_score",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="f_beta", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_soft_dice_loss",
                 confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="dice", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_soft_precision",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="precision", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_soft_recall",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_prob, metric="recall", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_hard_f1_score",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="f_beta", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_hard_precision",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="precision", average="samples"), on_step=False, on_epoch=True)
        self.log("train_sample_hard_recall",
                 1 - confusion_matrix_loss(y_true=y_true, y_pred=y_pred_label, metric="recall", average="samples"), on_step=False, on_epoch=True)

    def _log_validation_stats(self, y_true, y_pred_logit):
        y_pred_prob = torch.sigmoid(y_pred_logit)
        y_pred_label = (y_pred_prob > self.threshold).float()

        y_true = (y_true > self.threshold).float()

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

