"""
precompute_stain.py — apply Macenko stain normalization to BreakHis once, offline.

Normalizing on the fly costs ~20 ms/image every epoch; doing it once and caching
the result makes training free of that cost. Images are written at the model's
input size, mirroring the source folder structure so patient/label parsing (and
therefore the patient-level split) is completely unchanged.

The stain reference is fitted on a REAL slide drawn from the TRAINING patients
only — never val/test — so no information leaks into evaluation.

Usage:
    python scripts/precompute_stain.py --config configs/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.utils.breakhis import _scan
from src.preprocessing.stain_normalization import StainNormalizationPipeline


def pick_reference(samples, train_patients, image_size, n_probe=60, seed=42):
    """Choose a 'typical' training slide: the one whose mean intensity is closest
    to the median of a random probe of training images (avoids picking an outlier)."""
    train_files = [f for f, _, pid in samples if pid in train_patients]
    rng = np.random.RandomState(seed)
    probe = [train_files[i] for i in rng.choice(len(train_files),
                                                min(n_probe, len(train_files)), replace=False)]
    imgs, means = [], []
    for f in probe:
        a = np.array(Image.open(f).convert("RGB").resize((image_size, image_size), Image.BILINEAR))
        imgs.append(a); means.append(a.mean())
    means = np.array(means)
    idx = int(np.argmin(np.abs(means - np.median(means))))
    print(f"  reference slide: {probe[idx].name}  (mean intensity {means[idx]:.1f})")
    return imgs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: <breakhis_dir>_stain)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    src_root = Path(cfg.data.breakhis_dir)
    out_root = Path(args.out) if args.out else Path(str(src_root).rstrip("/\\") + "_stain")
    size = cfg.data.image_size

    print(f"Source : {src_root}")
    print(f"Output : {out_root}")

    samples = _scan(src_root, cfg.data.get("magnification", None))
    print(f"Found {len(samples)} images")

    # Reproduce the exact patient-level split so the reference is train-only.
    patients = sorted({pid for _, _, pid in samples})
    tv, _test = train_test_split(patients, test_size=0.15, random_state=cfg.project.seed)
    train_p, _val = train_test_split(tv, test_size=0.15 / (1 - 0.15), random_state=cfg.project.seed)
    train_p = set(train_p)
    print(f"Fitting stain reference on training patients only ({len(train_p)} patients)")

    ref = pick_reference(samples, train_p, size, seed=cfg.project.seed)
    pipe = StainNormalizationPipeline("macenko", reference_image=ref)

    changed = unchanged = 0
    for i, (path, _lbl, _pid) in enumerate(samples, 1):
        rel = Path(path).relative_to(src_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
        norm = pipe.normalize(arr)
        # normalize() returns the input unchanged if Macenko fails on that image
        if np.array_equal(norm, arr):
            unchanged += 1
        else:
            changed += 1
        Image.fromarray(norm).save(dst)

        if i % 500 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  normalized={changed}  fallback={unchanged}")

    print(f"\nDone. {changed} normalized, {unchanged} left unchanged (Macenko fallback).")
    if unchanged > 0.05 * len(samples):
        print("WARNING: >5% fell back — check the stain normalizer before training.")
    print(f"\nNext: set  data.breakhis_dir: \"{out_root.as_posix()}\"  in {args.config}")


if __name__ == "__main__":
    main()
