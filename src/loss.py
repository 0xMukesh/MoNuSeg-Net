import torch
from torch import nn
import segmentation_models_pytorch as smp
from einops import rearrange


class FocalDiceLoss(nn.Module):
    def __init__(
        self, num_classes: int, alpha: float = 0.25, beta: float = 0.5, gamma: float = 2
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            log_loss=True,
            from_logits=True,
        )
        self.focal = smp.losses.FocalLoss(mode="multiclass")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = rearrange(targets, "b 1 h w -> b h w")
        focal_loss = self.focal(preds, targets)
        dice_loss = self.dice(preds, targets)

        return (1 - self.beta) * focal_loss + self.beta * dice_loss
