"""
Test-Time Augmentation (TTA)
Average predictions over N augmented views at inference.
Typically adds 0.3-0.5% accuracy on top of the base model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.augmentations import get_tta_transforms


class TTAEvaluator:
    """
    Evaluates a model using test-time augmentation.
    For each image, generates n_augments views, runs inference,
    and averages the softmax probabilities.
    """
    def __init__(
        self,
        model,
        device: torch.device,
        n_augments: int = 10,
        image_size: int = 224,
    ):
        self.model = model
        self.device = device
        self.transforms = get_tta_transforms(image_size, n_augments)
        self.n_augments = len(self.transforms)

    @torch.no_grad()
    def predict_batch(self, images_np: List[np.ndarray]) -> torch.Tensor:
        """
        Args:
            images_np: list of [H, W, 3] numpy images
        Returns:
            probs: [B, 2] averaged probabilities
        """
        import torch
        B = len(images_np)
        all_probs = torch.zeros(B, 2)

        for transform in self.transforms:
            tensors = []
            for img in images_np:
                t = transform(image=img)["image"]
                tensors.append(t)
            batch = torch.stack(tensors).to(self.device)

            with torch.cuda.amp.autocast(enabled=True):
                out = self.model(batch)

            probs = F.softmax(out["logits"], dim=1).cpu()
            all_probs += probs

        return all_probs / self.n_augments

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        """Run TTA evaluation over entire dataloader."""
        from ..evaluation.metrics import compute_metrics

        self.model.eval()
        all_probs, all_labels = [], []

        for batch in tqdm(dataloader, desc="TTA Evaluation"):
            images_np = batch["image_np"]
            labels = batch["label"]

            # Convert list of arrays
            if isinstance(images_np, torch.Tensor):
                images_np = [images_np[i].numpy() for i in range(images_np.shape[0])]

            probs = self.predict_batch(images_np)
            all_probs.append(probs)
            all_labels.append(labels)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        # Convert probs to logits for metric computation
        logits = torch.log(all_probs.clamp(min=1e-8))
        return compute_metrics(logits, all_labels)
