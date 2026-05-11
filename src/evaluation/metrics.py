import torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              matthews_corrcoef, confusion_matrix, roc_curve)

def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.softmax(logits, dim=1)[:,1].numpy()
    preds = (probs >= threshold).astype(int)
    labels_np = labels.numpy()
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0,1]).ravel()
    return {
        "accuracy":    accuracy_score(labels_np, preds),
        "auc":         roc_auc_score(labels_np, probs) if len(set(labels_np)) > 1 else 0.0,
        "sensitivity": tp / (tp+fn+1e-8),
        "specificity": tn / (tn+fp+1e-8),
        "f1":          f1_score(labels_np, preds, zero_division=0),
        "mcc":         matthews_corrcoef(labels_np, preds),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def print_evaluation_report(metrics, title="Evaluation"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  AUC         : {metrics['auc']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1 Score    : {metrics['f1']:.4f}")
    print(f"  MCC         : {metrics['mcc']:.4f}")
    print(f"  TP={metrics['tp']:.0f}  TN={metrics['tn']:.0f}  "
          f"FP={metrics['fp']:.0f}  FN={metrics['fn']:.0f}")
    print(f"{'='*50}")


def find_optimal_threshold(logits, labels, optimize_for="sensitivity", min_specificity=0.90):
    """Find optimal classification threshold."""
    probs = torch.softmax(logits, dim=1)[:,1].numpy()
    labels_np = labels.numpy()

    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    specificity = 1 - fpr

    if optimize_for == "sensitivity":
        # Find threshold that maximizes sensitivity with specificity >= min_specificity
        valid = specificity >= min_specificity
        if valid.any():
            best_idx = np.argmax(tpr[valid])
            indices = np.where(valid)[0]
            return float(thresholds[indices[best_idx]])

    # Default: Youden's J statistic
    j_scores = tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def plot_roc_curve(logits, labels, save_path="results/roc_curve.png"):
    """Plot and save ROC curve."""
    probs = torch.softmax(logits, dim=1)[:,1].numpy()
    labels_np = labels.numpy()

    fpr, tpr, _ = roc_curve(labels_np, probs)
    auc = roc_auc_score(labels_np, probs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — CancerNet', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved to {save_path}")