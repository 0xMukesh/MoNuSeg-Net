import torch
from torch import nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import random
from typing import Literal


class CombinedTransform:
    def __init__(
        self,
        rotation_degrees: float = 0.0,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        img_additional_transform=None,
        mask_additional_transform=None,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.img_additional_transform = img_additional_transform
        self.mask_additional_transform = mask_additional_transform

    def __call__(self, img, mask):
        img, mask = self._apply_synchronized_transforms(img, mask)

        if self.img_additional_transform:
            img = self.img_additional_transform(img)

        if self.mask_additional_transform:
            mask = self.mask_additional_transform(mask)

        return img, mask

    def _apply_synchronized_transforms(self, img, mask):
        if random.random() < self.h_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < self.v_flip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if self.rotation_degrees > 0:
            degrees = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = TF.rotate(img, degrees, fill=[0.0])
            mask = TF.rotate(mask, degrees, fill=[0.0])

        return img, mask


def run_inference(model: nn.Module, loader: DataLoader, device: Literal["cuda", "cpu"]):
    model.eval()
    avg_dice_score = 0.0

    dice = smp.losses.DiceLoss(
        mode="multiclass",
        log_loss=True,
        from_logits=True,
    )

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device).float(), mask.to(device).float()
            preds = model(img)
            avg_dice_score += 1 - dice(preds, mask.long())

    return avg_dice_score / len(loader)
