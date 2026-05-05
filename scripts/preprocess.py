"""
preprocess.py — Verify dataset structure and run stain normalization
Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --verify-only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from omegaconf import OmegaConf

from src.preprocessing.stain_normalization import StainNormalizationPipeline


def verify_dataset(raw_dir: str) -> dict:
    """Check dataset structure and count samples."""
    root = Path(raw_dir)
    if not root.exists():
        print(f"ERROR: {raw_dir} does not exist.")
        print("Please download the IDC dataset from Kaggle first:")
        print("  kaggle datasets download -d paultimothymooney/breast-histopathology-images")
        print("  unzip breast-histopathology-images.zip -d data/raw/")
        return {}

    stats = {"patients": 0, "benign": 0, "malignant": 0, "total": 0}
    for patient_dir in root.iterdir():
        if not patient_dir.is_dir():
            continue
        stats["patients"] += 1
        for label, key in [(0, "benign"), (1, "malignant")]:
            label_dir = patient_dir / str(label)
            if label_dir.exists():
                count = len(list(label_dir.glob("*.png")))
                stats[key] += count
                stats["total"] += count

    print(f"\nDataset Statistics:")
    print(f"  Patients:   {stats['patients']:,}")
    print(f"  Benign:     {stats['benign']:,} patches")
    print(f"  Malignant:  {stats['malignant']:,} patches")
    print(f"  Total:      {stats['total']:,} patches")
    print(f"  Class ratio: {stats['malignant']/max(stats['total'],1)*100:.1f}% malignant")
    return stats


def run_stain_normalization(raw_dir: str, processed_dir: str, max_samples: int = None):
    """Apply stain normalization and save to processed dir."""
    root = Path(raw_dir)
    out_root = Path(processed_dir)
    normalizer = StainNormalizationPipeline(method="macenko")

    all_imgs = list(root.rglob("*.png"))
    if max_samples:
        all_imgs = all_imgs[:max_samples]

    print(f"\nNormalizing {len(all_imgs)} images...")
    errors = 0

    for img_path in tqdm(all_imgs):
        # Recreate directory structure
        rel_path = img_path.relative_to(root)
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            normalized = normalizer.normalize(img_rgb)
            normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), normalized_bgr)
        except Exception as e:
            errors += 1
            # Copy original if normalization fails
            shutil.copy2(img_path, out_path)

    print(f"Done. Errors: {errors}/{len(all_imgs)}")
    print(f"Processed images saved to: {processed_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for testing (None = all)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Step 1: Verify
    stats = verify_dataset(cfg.data.raw_dir)
    if not stats or args.verify_only:
        return

    if stats["total"] == 0:
        print("No images found. Check dataset structure.")
        return

    # Step 2: Stain normalization
    if not args.no_normalize:
        run_stain_normalization(
            cfg.data.raw_dir,
            cfg.data.processed_dir,
            max_samples=args.max_samples,
        )
        print(f"\nPreprocessing complete.")
        print(f"Update config 'data.raw_dir' to '{cfg.data.processed_dir}' to use normalized images.")
    else:
        print("Skipping normalization (--no-normalize).")


if __name__ == "__main__":
    main()
