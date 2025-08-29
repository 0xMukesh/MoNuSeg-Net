import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.segmentation.dice import DiceScore
from einops import rearrange


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.9, gamma: float = 2) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.ndim > 2:
            preds = rearrange(preds, "b c h w -> (b h w) c")
            targets = rearrange(targets, "b c h w -> (b h w) c")

        logpt = F.log_softmax(preds, dim=1)
        pt = logpt.exp()

        logpt = logpt.gather(dim=1, index=targets.type(torch.int32)).squeeze(1)
        pt = pt.gather(dim=1, index=targets.type(torch.int64)).squeeze(1)

        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self, num_classes: int, alpha: float = 0.9, beta: float = 0.5, gamma: float = 2
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        self.dice = DiceScore(num_classes, average="micro")
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.beta) * self.focal(preds, targets) + self.beta * self.dice(
            preds, targets
        )
