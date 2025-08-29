import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.segmentation.dice import DiceScore
from einops import rearrange
from typing import Literal


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, gamma: float = 2) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        targets = rearrange(targets, "b 1 h w -> b h w").type(torch.long)
        ce_loss = F.cross_entropy(input=preds, target=targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class FocalDiceLoss(nn.Module):
    def __init__(
        self, num_classes: int, alpha: float = 0.25, beta: float = 0.5, gamma: float = 2
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        self.dice = DiceScore(num_classes, average="macro", input_format="index")
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(preds, targets)

        targets = targets.squeeze(1).long()
        targets = F.one_hot(targets, num_classes=5)
        targets = targets.permute(0, 3, 1, 2)

        preds = torch.softmax(preds, dim=1).long()
        dice_score = self.dice(preds, targets)
        dice_loss = 1 - dice_score

        return (1 - self.beta) * focal_loss + self.beta * dice_loss


if __name__ == "__main__":
    preds = torch.floor(torch.randn(8, 5, 256, 256) * 5)
    targets = torch.randint(0, 5, (8, 1, 256, 256)).type(torch.float)
    criterion = FocalDiceLoss(num_classes=5)
    loss = criterion(preds, targets)
    print(loss)
