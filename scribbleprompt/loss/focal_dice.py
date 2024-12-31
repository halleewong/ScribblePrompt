
from typing import Tuple, Optional, Union, List, Literal

import torch
from torch import Tensor
import torch.nn as nn

from pylot.metrics.util import _metric_reduction, _inputs_as_onehot
from pylot.loss.segmentation import soft_dice_loss


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Soft Dice Loss
    """
    def __init__(self, from_logits: bool = False, gamma: float = 20.0, batch_reduction: Optional[Literal["mean"]] = None, **kwargs):
        super().__init__()
        self.batch_reduction = batch_reduction
        self.from_logits = from_logits
        self.gamma = gamma
        self.kwargs = kwargs

    def __call__(self, y_pred, y_true, ):
        # y_pred shape: B x 1 x H x W
        # We are doing binary segmentation so channel = 1

        focal_loss_term = focal_loss(y_pred, y_true, 
                                     gamma=self.gamma, 
                                     reduction='mean', 
                                     batch_reduction=None,
                                     from_logits=self.from_logits,
                                     )

        dice_loss_term = soft_dice_loss(y_pred, y_true,
                              mode='binary',
                              weights=None,
                              reduction='mean', # there's only 1 channel so this is fine
                              batch_reduction=None,
                              from_logits=self.from_logits,
                              **self.kwargs
                              )
        
        loss = focal_loss_term + dice_loss_term

        if self.batch_reduction == 'mean':
            return loss.mean()
        else:
            return loss

# -----------------------------------------------------------------------------
# Focal Loss
# -----------------------------------------------------------------------------

def focal_loss(
    y_pred: Tensor,
    y_true: Tensor,
    gamma: float = 20.0,
    weights: Optional[Tensor] = None,
    channel_weights: Optional[Tensor] = None,
    mode: str = "auto",
    reduction: str = "mean", # Reduction over channels
    batch_reduction: str = "mean", 
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    eps: float = 1e-7
) -> Tensor:
    """
    Binary focall loss that allows per-pixel weights
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    """
    if weights is not None:
        batch_size, num_classes = y_pred.shape[:2]
        weights = weights.view(batch_size, num_classes, -1)

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )

    loss = - binary_focal_cross_entropy(
        y_pred, y_true, weights=weights, eps=eps, dim=-1, gamma=gamma
    )

    batch_loss = _metric_reduction(
        loss, reduction=reduction, batch_reduction=batch_reduction, weights=channel_weights, ignore_index=ignore_index
    )

    return batch_loss

def binary_focal_cross_entropy(
    y_pred: Tensor,
    y_true: Tensor,
    weights: Optional[Tensor] = None,
    gamma: float = 20.0,
    eps: float = 1e-7,
    dim = None,
    ):
    """
    Returns -binary focal loss
    https://focal-loss.readthedocs.io/en/latest/generated/focal_loss.binary_focal_loss.html#focal_loss.binary_focal_loss
    """
    assert y_pred.shape == y_true.shape, f"y_pred.shape {y_pred.shape} != y_true.shape {y_true.shape}"
    if weights is not None:
        assert y_pred.shape == weights.shape, f"y_pred.shape={y_pred.shape}, weights.shape={weights.shape} do not match"

    left_term = y_true * torch.log(y_pred + eps) * (1 - y_pred)**gamma
    right_term = (1 - y_true) * torch.log(1-y_pred + eps) * y_pred**gamma

    if weights is not None:
        return torch.mean((left_term + right_term)*weights, dim=dim)
    else:
        return torch.mean(left_term + right_term, dim=dim)
    
