"""
BreakHis dataloader with patient-level splits.

Expects the extracted BreaKHis_v1 dataset, whose canonical layout is:
    <root>/histology_slides/breast/{benign,malignant}/SOB/<subtype>/<patient>/<mag>X/*.png
with filenames like  SOB_B_TA-14-4659-40-001.png  (B=benign / M=malignant, patient '14-4659').

The scanner is tolerant: it recursively finds every .png, infers the label from the
path/filename, and parses the patient id from the filename (folder fallback), so it also
works with flattened Kaggle mirrors of the dataset.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def _parse_label_patient(path: Path):
    """Return (label, patient_id). label: 1=malignant, 0=benign, None=unknown."""
    p = str(path).lower()
    name = path.name.lower()
    if "malignant" in p or "_m_" in name:
        label = 1
    elif "benign" in p or "_b_" in name:
        label = 0
    else:
        label = None

    # SOB_B_TA-14-4659-40-001 -> patient '14-4659'
    parts = path.stem.split("-")
    if len(parts) >= 4:
        patient = f"{parts[-4]}-{parts[-3]}"
    else:
        patient = path.parent.parent.name  # fallback: parent of the magnification folder
    return label, patient


def _scan(root, magnification=None):
    root = Path(root)
    files = sorted(root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(
            f"No .png images found under {root}. Extract BreaKHis_v1 there — expected "
            f".../breast/{{benign,malignant}}/SOB/<subtype>/<patient>/<mag>X/*.png")
    samples = []
    for f in files:
        label, pid = _parse_label_patient(f)
        if label is None:
            continue
        if magnification is not None and str(magnification) not in f.parent.name:
            continue
        samples.append((f, label, pid))
    if not samples:
        raise RuntimeError(f"Found {len(files)} PNGs under {root} but classified none — "
                           f"check the folder layout / magnification filter.")
    return samples


class BreakHisDataset(Dataset):
    def __init__(self, samples, split="train", transform=None, image_size=224):
        self.samples = samples          # list of (path, label, patient_id)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        n_mal = sum(s[1] for s in samples)
        print(f"[{split}] {len(samples)} images | {n_mal} malignant | "
              f"{len(samples) - n_mal} benign")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img)
        if self.transform:
            image = self.transform(image=img_np)["image"]
        else:
            image = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return {"image": image,
                "label": torch.tensor(label, dtype=torch.long),
                "patient_id": pid}


def get_breakhis_dataloaders(root_dir, train_transform, val_transform,
                             batch_size=32, num_workers=4, seed=42, image_size=224,
                             magnification=None, val_ratio=0.15, test_ratio=0.15, **kwargs):
    samples = _scan(root_dir, magnification)

    patients = sorted({pid for _, _, pid in samples})
    if len(patients) < 3:
        raise RuntimeError(f"Only {len(patients)} patient id(s) parsed — cannot make a "
                           f"patient-level split. Check the dataset layout.")
    tv, test_p = train_test_split(patients, test_size=test_ratio, random_state=seed)
    train_p, val_p = train_test_split(tv, test_size=val_ratio / (1 - test_ratio),
                                      random_state=seed)
    train_p, val_p, test_p = set(train_p), set(val_p), set(test_p)
    print(f"Patient split — train:{len(train_p)} val:{len(val_p)} test:{len(test_p)} "
          f"(of {len(patients)} patients)")

    def subset(pset):
        return [s for s in samples if s[2] in pset]

    train_ds = BreakHisDataset(subset(train_p), "train", train_transform, image_size)
    val_ds   = BreakHisDataset(subset(val_p),   "val",   val_transform,   image_size)
    test_ds  = BreakHisDataset(subset(test_p),  "test",  val_transform,   image_size)

    use_pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin)
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)
    return train_loader, val_loader, test_loader
