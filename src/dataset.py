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
        in_img_path = self.img_files[idx]
        out_img_path = self.img_files[idx].replace("in.png", "out.png")

        in_img = Image.open(in_img_path)
        out_img = Image.open(out_img_path)

        if self.transform:
            in_img, out_img = self.transform(in_img, out_img)

        return in_img, out_img
