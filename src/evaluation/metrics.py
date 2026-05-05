import torch, numpy as np
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