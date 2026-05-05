"""
evaluate.py — Load checkpoint and run full evaluation
Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --tta
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --gradcam
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path

from src.models.cancernet import CancerNet
from src.utils.dataset import get_dataloaders
from src.utils.augmentations import get_train_transforms, get_val_transforms
from src.utils.seed import set_seed
from src.evaluation.metrics import (
    compute_metrics, print_evaluation_report,
    find_optimal_threshold, plot_roc_curve,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tta", action="store_true", help="Use TTA")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM samples")
    parser.add_argument("--n-gradcam", type=int, default=10)
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    set_seed(cfg.project.seed)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = CancerNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val AUC={ckpt.get('metrics', {}).get('auc', '?'):.4f}")

    # Dataloaders
    _, _, test_loader = get_dataloaders(
        root_dir=cfg.data.raw_dir,
        train_transform=get_train_transforms(cfg.data.image_size),
        val_transform=get_val_transforms(cfg.data.image_size),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.project.seed,
    )

    # Standard evaluation
    print("\nRunning standard evaluation...")
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            out = model(images)
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["label"])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Standard metrics
    metrics = compute_metrics(all_logits, all_labels)
    print_evaluation_report(metrics, "Standard Evaluation")

    # Optimal threshold
    opt_thresh = find_optimal_threshold(all_logits, all_labels, optimize_for="sensitivity")
    print(f"Optimal threshold (max sensitivity @ specificity>=90%): {opt_thresh:.3f}")
    opt_metrics = compute_metrics(all_logits, all_labels, threshold=opt_thresh)
    print_evaluation_report(opt_metrics, f"Optimal Threshold ({opt_thresh:.3f})")

    # ROC curve
    os.makedirs(cfg.logging.results_dir, exist_ok=True)
    plot_roc_curve(all_logits, all_labels, f"{cfg.logging.results_dir}/roc_curve.png")

    # TTA evaluation
    if args.tta:
        print("\nRunning TTA evaluation...")
        from src.evaluation.tta import TTAEvaluator
        tta_eval = TTAEvaluator(model, device, n_augments=cfg.augmentation.tta.n_augments)
        tta_metrics = tta_eval.evaluate(test_loader)
        print_evaluation_report(tta_metrics, "TTA Evaluation")

    # Grad-CAM visualization
    if args.gradcam:
        print(f"\nGenerating {args.n_gradcam} Grad-CAM visualizations...")
        from src.evaluation.gradcam import save_gradcam_visualization
        gradcam_dir = Path(cfg.logging.results_dir) / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for batch in test_loader:
            for i in range(len(batch["label"])):
                if count >= args.n_gradcam:
                    break
                img_tensor = batch["image"][i]
                img_np = batch["image_np"][i]
                label = batch["label"][i].item()
                if isinstance(img_np, torch.Tensor):
                    img_np = img_np.numpy()

                save_gradcam_visualization(
                    model, img_tensor, img_np, label,
                    save_path=str(gradcam_dir / f"sample_{count:03d}_label{label}.png"),
                    device=device,
                )
                count += 1
            if count >= args.n_gradcam:
                break
        print(f"Grad-CAM images saved to {gradcam_dir}/")


if __name__ == "__main__":
    main()
