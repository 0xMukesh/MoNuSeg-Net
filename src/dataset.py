from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List


class MoNuSACDataset(Dataset):
    def __init__(self, root: str, transform=None) -> None:
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
        in_img = self.img_files[idx]
        img_dir = in_img.split("in.png")[0]
        mask_imgs = []

        img = Image.open(self.img_files[idx])
        if self.transform:
            img = self.transform(img)

        for i in range(4):
            mask_img = Image.open(os.path.join(img_dir, f"{i}.png"))
            if self.transform:
                mask_img = self.transform(mask_img)

            mask_imgs.append(mask_img)

        return img, mask_imgs
