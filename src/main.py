import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataclasses import dataclass

from .models import AttentionResidualUNet
from .dataset import MoNuSACDataset
from .utils import CombinedTransform


@dataclass
class TrainingConfig:
    target_size = (512, 512)
    batch_size = 6
    pin_memory = True
    num_workers = 2

    epochs = 1
    lr = 1e-4

    train_dataset_root_dir = "./data/train/masks"
    test_dataset_root_dir = "./data/test/masks"
    checkpoints_dir = "checkpoints"


config = TrainingConfig()


additional_transform = T.Compose([T.Resize(config.target_size), T.ToTensor()])

train_transform = CombinedTransform(
    rotation_degrees=15.0,
    img_additional_transform=additional_transform,
    mask_additional_transform=additional_transform,
)
test_transform = CombinedTransform(
    rotation_degrees=0,
    h_flip_prob=0,
    v_flip_prob=0,
    img_additional_transform=additional_transform,
    mask_additional_transform=additional_transform,
)

train_dataset = MoNuSACDataset(
    root=config.train_dataset_root_dir, transform=train_transform
)
test_dataset = MoNuSACDataset(
    root=config.test_dataset_root_dir, transform=test_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=config.pin_memory,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=config.pin_memory,
    num_workers=config.num_workers,
)

for epoch in range(config.epochs):
    """"""
