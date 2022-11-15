import torch


def soft_dice_loss(y_true, y_pred, epsilon=1e-6, reduction="sum"):
    """
    Compute soft dice loss according to V-Net paper.

    For each channel in each sample, a scalar dice loss is calculated.
    The total dice loss for the batch is reduced
    over the sample and channel axes.

    Parameters
    ----------
    y_true : torch.Tensor
        Has shape [N, C, ...]
    y_pred : torch.Tensor
        Has shape [N, C, ...]
    epsilon : float
        Small constant to be the minimum denominator;
        prevents numerical instability
    reduction : str
        One of "sum", "mean", "none";
        how the [N] loss list per data point is reduced.

    Returns
    -------
    loss : torch.Tensor
    """
    # axes to sum over, ignoring the batch and channel dimensions
    axes = tuple(range(2, y_true.ndim))     # sum over spatial dimensions
    numerator = 2 * (y_pred * y_true).sum(axes)
    denominator = (y_pred ** 2).sum(axes) + (y_true ** 2).sum(axes)
    # sum over channel axis; use (1 - Dice) as loss
    loss_list = (1 - numerator / denominator.clamp(min=epsilon)).sum(1)
    # reduce over sample axis
    if reduction == "mean":
        return loss_list.mean()
    elif reduction == "sum":
        return loss_list.sum()
    elif reduction == "none":
        return loss_list
    else:
        raise ValueError(f"Unknown reduction keyword {reduction}")




