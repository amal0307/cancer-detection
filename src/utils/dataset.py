import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple


class IDCDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None,
                 patient_ids=None, image_size=224, use_gnn=True):
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.transform  = transform
        self.image_size = image_size
        self.use_gnn    = use_gnn
        self.samples    = []  # (path, label, patient_id)

        for patient_dir in sorted(self.root_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name
            if patient_ids is not None and pid not in patient_ids:
                continue
            for label in [0, 1]:
                label_dir = patient_dir / str(label)
                if not label_dir.exists():
                    continue
                for img_path in label_dir.glob("*.png"):
                    self.samples.append((img_path, label, pid))

        print(f"[{split}] {len(self.samples)} patches | "
              f"{sum(s[1] for s in self.samples)} malignant | "
              f"{sum(1-s[1] for s in self.samples)} benign")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]

        # Load and force resize to consistent size
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img)  # always [image_size, image_size, 3]

        if self.transform:
            image = self.transform(image=img_np)["image"]
        else:
            image = torch.tensor(
                img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        result = {
            "image":      image,
            "label":      torch.tensor(label, dtype=torch.long),
            "patient_id": pid,
        }

        # Only compute image_np when GNN is enabled (cv2.resize is expensive)
        if self.use_gnn:
            result["image_np"] = cv2.resize(img_np, (50, 50))

        return result


def get_patient_level_splits(root_dir, val_ratio=0.15, test_ratio=0.15, seed=42, max_patients=None):
    root = Path(root_dir)
    all_patients = sorted([d.name for d in root.iterdir() if d.is_dir()])

    if not all_patients:
        raise ValueError(f"No patient directories found in {root_dir}")

    # Optionally limit the number of patients for faster training
    if max_patients is not None and max_patients < len(all_patients):
        import random
        rng = random.Random(seed)
        all_patients = sorted(rng.sample(all_patients, max_patients))
        print(f"Using {max_patients}/{len(all_patients)} patients (subset mode)")

    train_val, test = train_test_split(
        all_patients, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_ratio / (1 - test_ratio), random_state=seed)

    print(f"Patient split — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


def get_dataloaders(root_dir, train_transform, val_transform,
                    batch_size=32, num_workers=0, seed=42, image_size=224,
                    max_patients=None, use_gnn=True):

    train_ids, val_ids, test_ids = get_patient_level_splits(root_dir, seed=seed, max_patients=max_patients)

    train_ds = IDCDataset(root_dir, "train", train_transform, train_ids, image_size, use_gnn=use_gnn)
    val_ds   = IDCDataset(root_dir, "val",   val_transform,   val_ids,   image_size, use_gnn=use_gnn)
    test_ds  = IDCDataset(root_dir, "test",  val_transform,   test_ids,  image_size, use_gnn=use_gnn)

    # pin_memory only works with CUDA
    use_pin = torch.cuda.is_available()

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    # Enable prefetching and persistent workers for faster data loading
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **common_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: IDCDataset) -> torch.Tensor:
    labels  = [s[1] for s in dataset.samples]
    counts  = torch.tensor([labels.count(0), labels.count(1)], dtype=torch.float)
    weights = 1.0 / counts
    return weights / weights.sum()