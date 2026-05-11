import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional

from ..models.cancernet import CancerNet
from .losses import get_loss
from ..evaluation.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: CancerNet,
        cfg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.loss_fn = get_loss(cfg).to(device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # AMP only on CUDA
        self.use_amp = cfg.training.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)

        self.best_auc = 0.0
        self.patience_counter = 0
        self.global_step = 0

        Path(cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.logging.results_dir).mkdir(parents=True, exist_ok=True)

        self._init_logging()

        # --- Performance optimizations ---

        # torch.compile() for fused kernels (PyTorch 2.0+)
        if cfg.training.get("compile_model", False) and hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model)
                print("  ✅ Model compiled successfully")
            except Exception as e:
                print(f"  ⚠️ torch.compile() failed, running without: {e}")

        # Enable TF32 on Ampere+ GPUs (A100, H100) for faster matmul
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("  ✅ TF32 and cuDNN benchmark enabled")

        # GNN graph cache to avoid recomputing SLIC every epoch
        self._graph_cache = {}
        self._use_graph_cache = cfg.graph.get("cache_graphs", False)
        if self._use_graph_cache:
            print("  ✅ GNN graph caching enabled")

    def _build_optimizer(self):
        cfg = self.cfg.training
        lr = cfg.optimizer.lr
        factor = cfg.optimizer.differential_lr.backbone_lr_factor

        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "cnn_branch.backbone" in name or "vit_branch.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * factor},
            {"params": head_params,     "lr": lr},
        ]
        return torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.optimizer.weight_decay,
        )

    def _build_scheduler(self):
        cfg = self.cfg.training
        total_steps = cfg.epochs * len(self.train_loader) // cfg.accumulation_steps
        warmup_steps = cfg.scheduler.warmup_epochs * len(self.train_loader) // cfg.accumulation_steps
        min_ratio = cfg.scheduler.min_lr / cfg.optimizer.lr

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_ratio, 0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_logging(self):
        self.use_wandb = False
        print("WandB disabled — logging to console only.")

    def _log(self, metrics: Dict, step: int):
        pass  # wandb disabled

    def _build_graph_batch(self, batch, batch_idx):
        """Build GNN graphs with optional caching to avoid repeated SLIC computation."""
        if not self.model.use_gnn:
            return None

        # Check cache first
        if self._use_graph_cache and batch_idx in self._graph_cache:
            return self._graph_cache[batch_idx].to(self.device)

        from ..preprocessing.graph_builder import build_batch_graphs

        try:
            images_np = batch["image_np"]
            if isinstance(images_np, torch.Tensor):
                images_np = [images_np[j].numpy() for j in range(images_np.shape[0])]
            graph_batch = build_batch_graphs(
                images_np,
                n_neighbors=self.cfg.graph.n_neighbors,
            )
            # Cache on CPU to save GPU memory
            if self._use_graph_cache:
                self._graph_cache[batch_idx] = graph_batch.cpu()
            return graph_batch.to(self.device)
        except Exception:
            return None

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []
        accumulation_steps = self.cfg.training.accumulation_steps

        self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        start = time.time()

        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Build GNN graph batch (cached after first epoch)
            graph_batch = self._build_graph_batch(batch, i)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):
                out = self.model(images, graph_batch)
                loss = self.loss_fn(out["logits"], labels)
                loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            total_loss += loss.item() * accumulation_steps
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(labels.cpu())

            if i % self.cfg.logging.log_every_n_steps == 0:
                elapsed = time.time() - start
                samples_per_sec = (i + 1) * self.cfg.training.batch_size / elapsed if elapsed > 0 else 0
                print(f"  Epoch {epoch} [{i}/{len(self.train_loader)}] "
                      f"loss={loss.item() * accumulation_steps:.4f} "
                      f"elapsed={elapsed:.1f}s "
                      f"({samples_per_sec:.0f} samples/s)")

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):
                out = self.model(images)
                loss = self.loss_fn(out["logits"], labels)

            total_loss += loss.item()
            all_logits.append(out["logits"].cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        path = Path(self.cfg.logging.checkpoint_dir)
        torch.save(ckpt, path / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(ckpt, path / "best_model.pth")
            print(f"  *** Saved best model (AUC={metrics.get('auc', 0):.4f}) ***")

    def train(self):
        cfg = self.cfg.training
        freeze_epochs = self.cfg.model.efficientnet.freeze_epochs

        print(f"\n{'='*60}")
        print(f"  CancerNet Training")
        print(f"  Device     : {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU        : {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP        : {self.use_amp}")
        print(f"  Epochs     : {cfg.epochs}")
        print(f"  Batch size : {cfg.batch_size}")
        print(f"  Train size : {len(self.train_loader.dataset)}")
        print(f"  Val size   : {len(self.val_loader.dataset)}")
        print(f"  GNN graphs : {'cached' if self._use_graph_cache else 'recomputed each epoch'}")
        print(f"{'='*60}\n")

        for epoch in range(1, cfg.epochs + 1):

            # Unfreeze backbones after freeze period
            if epoch == freeze_epochs + 1:
                print(f"\n--- Unfreezing backbones at epoch {epoch} ---\n")
                self.model.unfreeze_backbones()
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()

            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{cfg.epochs}")

            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate()

            epoch_time = time.time() - epoch_start

            print(f"  Train | loss={train_metrics['loss']:.4f}  "
                  f"acc={train_metrics['accuracy']:.4f}  "
                  f"auc={train_metrics['auc']:.4f}")
            print(f"  Val   | loss={val_metrics['loss']:.4f}  "
                  f"acc={val_metrics['accuracy']:.4f}  "
                  f"auc={val_metrics['auc']:.4f}  "
                  f"sens={val_metrics['sensitivity']:.4f}  "
                  f"spec={val_metrics['specificity']:.4f}")
            print(f"  Time  | {epoch_time:.1f}s "
                  f"(~{epoch_time * (cfg.epochs - epoch) / 60:.1f} min remaining)")

            val_auc  = val_metrics["auc"]
            is_best  = val_auc > self.best_auc

            if is_best:
                self.best_auc = val_auc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  No improvement for "
                      f"{self.patience_counter}/{cfg.early_stopping.patience} epochs")

            self.save_checkpoint(epoch, val_metrics, is_best)

            if self.patience_counter >= cfg.early_stopping.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete.")
        print(f"  Best validation AUC : {self.best_auc:.4f}")
        print(f"  Checkpoint saved to : "
              f"{self.cfg.logging.checkpoint_dir}/best_model.pth")
        print(f"{'='*60}\n")