import csv, json, logging
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Persists per-epoch history (CSV + JSON), a text log, and metrics JSON."""

    def __init__(self, log_dir="logs", results_dir="results", run_name=None):
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(results_dir); self.results_dir.mkdir(parents=True, exist_ok=True)

        self.history = []
        self.history_csv = self.log_dir / f"{self.run_name}_history.csv"
        self.history_json = self.log_dir / f"{self.run_name}_history.json"

        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False
        fh = logging.FileHandler(self.log_dir / f"{self.run_name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)
        self.info(f"Run '{self.run_name}' started")

    def info(self, msg):
        self.logger.info(msg)

    @staticmethod
    def _clean(d):
        return {k: (round(float(v), 6) if isinstance(v, (int, float, bool)) else v)
                for k, v in d.items()}

    def log_epoch(self, epoch, train_metrics, val_metrics, lr=None, epoch_time=None):
        row = {"epoch": epoch}
        row.update({f"train_{k}": round(float(v), 6) for k, v in train_metrics.items()})
        row.update({f"val_{k}": round(float(v), 6) for k, v in val_metrics.items()})
        if lr is not None: row["lr"] = round(float(lr), 8)
        if epoch_time is not None: row["epoch_time_s"] = round(float(epoch_time), 1)
        self.history.append(row)

        first = not self.history_csv.exists()
        with open(self.history_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if first: w.writeheader()
            w.writerow(row)
        with open(self.history_json, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        self.info(f"epoch {epoch} | train_auc={row.get('train_auc', float('nan')):.4f} "
                  f"val_auc={row.get('val_auc', float('nan')):.4f} "
                  f"val_acc={row.get('val_accuracy', float('nan')):.4f}")

    def save_metrics(self, metrics, name="metrics.json", extra=None):
        payload = {"run_name": self.run_name,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
        payload.update(self._clean(metrics))
        if extra: payload.update(self._clean(extra))
        out = self.results_dir / name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.info(f"Saved {out}")
        return out
