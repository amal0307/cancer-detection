import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from omegaconf import OmegaConf

from src.models.cancernet import CancerNet
from src.training.trainer import Trainer
from src.utils.dataset import get_dataloaders
from src.utils.augmentations import get_train_transforms, get_val_transforms
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.project.seed)

    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    max_patients = cfg.data.get("max_patients", None)

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=cfg.data.raw_dir,
        train_transform=get_train_transforms(cfg.data.image_size),
        val_transform=get_val_transforms(cfg.data.image_size),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.project.seed,
        image_size=cfg.data.image_size,
        max_patients=max_patients,
    )

    model = CancerNet(cfg)
    trainer = Trainer(model, cfg, train_loader, val_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()