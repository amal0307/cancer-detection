"""
tests/test_model.py
Unit tests to verify all model components work correctly.
Run: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


# ── Minimal config for testing ───────────────────────────────────────────────
@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {
            "efficientnet": {"backbone": "efficientnet_b0", "pretrained": False, "embedding_dim": 1280, "freeze_epochs": 1},
            "vit": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "embedding_dim": 192, "freeze_epochs": 1},
            "gnn": {"num_node_features": 64, "hidden_dim": 64, "output_dim": 64, "num_layers": 2, "dropout": 0.1},
            "fusion": {"input_dim": 512, "num_heads": 4, "temperature_learnable": True, "dropout": 0.1},
            "classifier": {"hidden_dims": [64], "num_classes": 2, "dropout": 0.1},
        },
        "training": {"mixed_precision": False},
    })


@pytest.fixture
def device():
    return torch.device("cpu")


# ── EfficientNet branch ───────────────────────────────────────────────────────
class TestEfficientNetBranch:
    def test_forward_shape(self):
        from src.models.efficientnet_branch import EfficientNetBranch
        model = EfficientNetBranch("efficientnet_b0", pretrained=False, embedding_dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_freeze_unfreeze(self):
        from src.models.efficientnet_branch import EfficientNetBranch
        model = EfficientNetBranch("efficientnet_b0", pretrained=False, embedding_dim=256)
        model.freeze_backbone()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0, "Backbone should have frozen params"
        model.unfreeze_backbone()
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen > 0, "Backbone should have unfrozen params after unfreeze"


# ── ViT branch ────────────────────────────────────────────────────────────────
class TestViTBranch:
    def test_forward_shape(self):
        from src.models.vit_branch import ViTBranch
        model = ViTBranch("vit_tiny_patch16_224", pretrained=False, embedding_dim=192)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 192), f"Expected (2, 192), got {out.shape}"


# ── GNN branch ────────────────────────────────────────────────────────────────
class TestGNNBranch:
    def test_forward_shape(self):
        pytest.importorskip("torch_geometric")
        from src.models.gnn_branch import GNNBranch
        from torch_geometric.data import Data, Batch

        model = GNNBranch(num_node_features=16, hidden_dim=32, output_dim=32, num_layers=2)
        graphs = []
        for _ in range(2):
            x = torch.randn(10, 16)
            edge_index = torch.randint(0, 10, (2, 20))
            graphs.append(Data(x=x, edge_index=edge_index))
        batch = Batch.from_data_list(graphs)
        out = model(batch)
        assert out.shape == (2, 32), f"Expected (2, 32), got {out.shape}"


# ── Fusion ────────────────────────────────────────────────────────────────────
class TestFusion:
    def test_bidirectional_cross_attention(self):
        from src.models.fusion import BidirectionalCrossAttentionFusion
        fusion = BidirectionalCrossAttentionFusion(
            cnn_dim=128, vit_dim=64, gnn_dim=32,
            fusion_dim=64, num_heads=4,
        )
        cnn = torch.randn(4, 128)
        vit = torch.randn(4, 64)
        gnn = torch.randn(4, 32)
        out = fusion(cnn, vit, gnn)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_attention_weights_stored(self):
        from src.models.fusion import BidirectionalCrossAttentionFusion
        fusion = BidirectionalCrossAttentionFusion(
            cnn_dim=128, vit_dim=64, gnn_dim=32, fusion_dim=64, num_heads=4
        )
        cnn = torch.randn(2, 128)
        vit = torch.randn(2, 64)
        gnn = torch.randn(2, 32)
        _ = fusion(cnn, vit, gnn)
        weights = fusion.get_attention_weights()
        assert "cnn->vit" in weights
        assert "vit->cnn" in weights


# ── Full CancerNet ─────────────────────────────────────────────────────────────
class TestCancerNet:
    def test_forward_output_keys(self, cfg):
        from src.models.cancernet import CancerNet
        model = CancerNet(cfg)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert "logits" in out
        assert "probs" in out
        assert out["logits"].shape == (2, 2)
        assert out["probs"].shape == (2, 2)
        assert torch.allclose(out["probs"].sum(dim=1), torch.ones(2), atol=1e-5)

    def test_return_features(self, cfg):
        from src.models.cancernet import CancerNet
        model = CancerNet(cfg)
        x = torch.randn(2, 3, 224, 224)
        out = model(x, return_features=True)
        assert "features" in out
        assert "cnn" in out["features"]
        assert "vit" in out["features"]

    def test_parameter_count(self, cfg):
        from src.models.cancernet import CancerNet
        model = CancerNet(cfg)
        counts = model.count_parameters()
        assert counts["total"] > 0
        print(f"\nTotal trainable parameters: {counts['total']:,}")


# ── Loss functions ────────────────────────────────────────────────────────────
class TestLosses:
    def test_focal_loss(self):
        from src.training.losses import FocalLoss
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_combined_loss(self):
        from src.training.losses import CombinedLoss
        loss_fn = CombinedLoss()
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_compute_metrics_keys(self):
        from src.evaluation.metrics import compute_metrics
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))
        metrics = compute_metrics(logits, labels)
        for key in ["accuracy", "auc", "sensitivity", "specificity", "f1", "mcc"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_perfect_predictions(self):
        from src.evaluation.metrics import compute_metrics
        # Simulate perfect predictions
        logits = torch.tensor([[10.0, 0.0]] * 50 + [[0.0, 10.0]] * 50)
        labels = torch.tensor([0] * 50 + [1] * 50)
        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == pytest.approx(1.0, abs=0.01)
        assert metrics["sensitivity"] == pytest.approx(1.0, abs=0.01)
        assert metrics["specificity"] == pytest.approx(1.0, abs=0.01)


# ── Stain normalization ────────────────────────────────────────────────────────
class TestStainNormalization:
    def test_normalize_returns_same_shape(self):
        from src.preprocessing.stain_normalization import StainNormalizationPipeline
        normalizer = StainNormalizationPipeline(method="macenko")
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = normalizer.normalize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_fallback_on_bad_image(self):
        from src.preprocessing.stain_normalization import StainNormalizationPipeline
        normalizer = StainNormalizationPipeline(method="macenko")
        bad_img = np.zeros((10, 10, 3), dtype=np.uint8)  # all black
        result = normalizer.normalize(bad_img)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
