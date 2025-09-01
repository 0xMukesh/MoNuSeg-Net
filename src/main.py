import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import os
from dataclasses import dataclass
from typing import cast

from .models import AttentionResidualUNet
from .dataset import MoNuSACPatchDataset
from .loss import FocalDiceLoss
from .utils import CombinedTransform, run_inference
from .constants import NAME_CLASS_MAPPING


@dataclass
class TrainingConfig:
    target_size = (256, 256)
    batch_size = 8
    pin_memory = True
    num_workers = 2

    num_classes = 1 + len(NAME_CLASS_MAPPING.values())
    epochs = 50
    lr = 1e-3

    weight_decay = 1e-4

    train_dataset_root_dir = "/content/data/train/masks"
    test_dataset_root_dir = "/content/data/test/masks"
    checkpoints_dir = "checkpoints"


config = TrainingConfig()


img_additional_transform = T.Compose(
    [
        T.ElasticTransform(),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize(mean=[0.6189], std=[0.1726]),
    ]
)
mask_additional_transform = T.Compose(
    [
        T.ElasticTransform(),
        T.Grayscale(),
        T.PILToTensor(),
    ]
)

train_transform = CombinedTransform(
    rotation_degrees=90.0,
    img_additional_transform=img_additional_transform,
    mask_additional_transform=mask_additional_transform,
)
test_transform = CombinedTransform(
    rotation_degrees=0,
    h_flip_prob=0,
    v_flip_prob=0,
    img_additional_transform=img_additional_transform,
    mask_additional_transform=mask_additional_transform,
)

train_dataset = MoNuSACPatchDataset(
    root=config.train_dataset_root_dir, transform=train_transform
)
test_dataset = MoNuSACPatchDataset(
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AttentionResidualUNet(in_channels=1, num_classes=config.num_classes).to(device)
loss_fn = FocalDiceLoss(num_classes=config.num_classes)
optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=config.epochs, eta_min=1e-6
)
scaler = GradScaler(device)

batch_losses = []
epoch_losses = []

best_dice_score = 0.0

for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, total=len(train_loader), leave=True)

    for img, mask in loop:
        img, mask = cast(torch.Tensor, img.to(device).float()), cast(
            torch.Tensor, mask.to(device).float()
        )

        with torch.autocast(device):
            pred = model(img)
            loss = loss_fn(pred, mask.long())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        batch_losses.append(loss.item())

        loop.set_description(f"epoch [{epoch+1}/{config.epochs}]")
        loop.set_postfix(loss=loss.item())

    scheduler.step()
    running_loss /= len(train_loader)

    epoch_losses.append(running_loss)
    val_dice_score = run_inference(model, test_loader, device)

    if val_dice_score > best_dice_score:
        os.makedirs(config.checkpoints_dir, exist_ok=True)

        best_dice_score = val_dice_score
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": running_loss,
            "dice_score": val_dice_score,
        }

        torch.save(
            checkpoint, os.path.join(config.checkpoints_dir, "best_dice_score.pth")
        )

    print(f"summary of epoch {epoch + 1}/{config.epochs}")
    print(f"  avg loss: {running_loss:.4f}")
    print(f"  val dice score: {val_dice_score:.4f}")
