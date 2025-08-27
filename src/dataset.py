import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Literal


class MoNuSACDataset(Dataset):
    def __init__(self, root: str, split: Literal["train", "test"]) -> None:
        super().__init__()

        self.root = root
        self.split = split
