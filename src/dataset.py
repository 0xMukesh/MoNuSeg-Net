from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from einops import rearrange
import os
from typing import List, Optional

from src.utils import PatchExtractor, CombinedTransform


class MoNuSACDataset(Dataset):
    def __init__(
        self, root: str, transform: Optional[CombinedTransform] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        self.img_files: List[str] = []

        for patient_dir in os.listdir(self.root):
            patient_dir = os.path.join(self.root, patient_dir)
            for img_dir in os.listdir(patient_dir):
                img_dir = os.path.join(patient_dir, img_dir)
                self.img_files.append(os.path.join(img_dir, "in.png"))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        mask_path = self.img_files[idx].replace("in.png", "out.png")

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


class MoNuSACPatchDataset(Dataset):
    def __init__(
        self,
        root: str,
        patch_size: int = 256,
        stride_size: int = 128,
        min_foreground_ratio: float = 0.1,
        transform: Optional[CombinedTransform] = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.transform = transform

        self.dataset = MoNuSACDataset(root, transform=None)
        self.patch_extractor = PatchExtractor(
            patch_size, stride_size, min_foreground_ratio
        )

        self.img_patches: List[Image.Image] = []
        self.mask_patches: List[Image.Image] = []

        self._generate_patches()

    def _generate_patches(self) -> None:
        for img, mask in self.dataset:
            img = np.array(img)
            mask = np.array(mask)

            img = rearrange(img, "h w c -> c h w")
            mask = rearrange(mask, "h w -> 1 h w")

            img_patches, mask_patches = self.patch_extractor.extract_patches_from_image(
                img, mask
            )

            for i, patch in enumerate(img_patches):
                ph, pw = self.patch_extractor._get_img_dims(patch)
                ih, iw = self.patch_extractor._get_img_dims(img)

                if ph != ih or pw != iw:
                    pad_h = max(0, self.patch_size - ph)
                    pad_w = max(0, self.patch_size - pw)

                    img_patches[i] = np.pad(
                        img_patches[i],
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )
                    mask_patches[i] = np.pad(
                        mask_patches[i],
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )

                    img_patches[i] = rearrange(img_patches[i], "c h w -> h w c")
                    mask_patches[i] = rearrange(mask_patches[i], "1 h w -> h w")

                self.img_patches.append(Image.fromarray(img_patches[i]))
                self.mask_patches.append(Image.fromarray(mask_patches[i]))

    def __len__(self) -> int:
        return len(self.img_patches)

    def __getitem__(self, idx):
        img_patch = self.img_patches[idx]
        mask_patch = self.mask_patches[idx]

        if self.transform:
            img_patch, mask_patch = self.transform(img_patch, mask_patch)

        return img_patch, mask_patch
