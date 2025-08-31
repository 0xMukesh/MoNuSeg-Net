import torch
from torch import nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from einops import parse_shape
import random
from typing import Literal, List


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


class PatchExtractor:
    def __init__(
        self,
        patch_size: int = 256,
        stride_size: int = 128,
        min_foreground_ratio: float = 0.1,
    ) -> None:
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.min_foreground_ratio = min_foreground_ratio

    def extract_patches_from_image(self, img: np.ndarray, mask: np.ndarray):
        img_patches: List[np.ndarray] = []
        mask_patches: List[np.ndarray] = []

        h, w = self._get_img_dims(img)

        for y in range(0, h - self.patch_size + 1, self.stride_size):
            for x in range(0, w - self.patch_size + 1, self.stride_size):
                if img.ndim == 3 and img.shape[0] <= 4:
                    img_patch = img[:, y : y + self.patch_size, x : x + self.patch_size]
                    mask_patch = mask[
                        :, y : y + self.patch_size, x : x + self.patch_size
                    ]
                else:
                    img_patch = img[y : y + self.patch_size, x : x + self.patch_size]
                    mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]

                if self._has_enough_foreground(mask_patch):
                    img_patches.append(img_patch)
                    mask_patches.append(mask_patch)

        return img_patches, mask_patches

    def _has_enough_foreground(self, mask: np.ndarray):
        h, w = self._get_img_dims(mask)
        non_zero_pixels = np.sum(mask > 0)
        ratio = non_zero_pixels / (h * w)

        return ratio >= self.min_foreground_ratio

    def _get_img_dims(self, img: np.ndarray):
        if img.ndim > 3:
            shape_dict = parse_shape(img, "n c h w")
            h, w = shape_dict["h"], shape_dict["w"]
        elif img.ndim == 3:
            if img.shape[0] <= 4:
                shape_dict = parse_shape(img, "c h w")
                h, w = shape_dict["h"], shape_dict["w"]
            else:
                shape_dict = parse_shape(img, "h w c")
                h, w = shape_dict["h"], shape_dict["w"]
        else:
            shape_dict = parse_shape(img, "h w")
            h, w = shape_dict["h"], shape_dict["w"]

        return h, w


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
