import torch, torch.nn as nn, torch.nn.functional as F
from .efficientnet_branch import EfficientNetBranch
from .vit_branch import ViTBranch
from .gnn_branch import GNNBranch
from .fusion import BidirectionalCrossAttentionFusion

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512,128], num_classes=2, dropout=0.4):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class CancerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cnn_branch = EfficientNetBranch(
            cfg.model.efficientnet.backbone, cfg.model.efficientnet.pretrained,
            cfg.model.efficientnet.embedding_dim)
        self.vit_branch = ViTBranch(
            cfg.model.vit.backbone, cfg.model.vit.pretrained,
            cfg.model.vit.embedding_dim)
        self.use_gnn = cfg.model.gnn.get("enabled", True)
        if self.use_gnn:
            try:
                self.gnn_branch = GNNBranch(
                    cfg.model.gnn.num_node_features, cfg.model.gnn.hidden_dim,
                    cfg.model.gnn.output_dim, cfg.model.gnn.num_layers, cfg.model.gnn.dropout)
            except Exception:
                self.use_gnn = False
                self.gnn_branch = None
                print("  ⚠️ GNN branch disabled (import failed)")
        else:
            self.gnn_branch = None
            print("  ⚡ GNN branch disabled via config (faster training)")

        fusion_dim = cfg.model.fusion.input_dim // 4
        self.fusion = BidirectionalCrossAttentionFusion(
            cfg.model.efficientnet.embedding_dim, cfg.model.vit.embedding_dim,
            cfg.model.gnn.output_dim, fusion_dim,
            cfg.model.fusion.num_heads, cfg.model.fusion.dropout)
        self.classifier = ClassificationHead(
            fusion_dim, cfg.model.classifier.hidden_dims,
            cfg.model.classifier.num_classes, cfg.model.classifier.dropout)
        self.cnn_branch.freeze_backbone()
        self.vit_branch.freeze_backbone()

    def unfreeze_backbones(self):
        self.cnn_branch.unfreeze_backbone()
        self.vit_branch.unfreeze_backbone()

    def forward(self, images, graph_batch=None, return_features=False):
        cnn_feat = self.cnn_branch(images)
        vit_feat = self.vit_branch(images)
        if graph_batch is not None and self.use_gnn:
            gnn_feat = self.gnn_branch(graph_batch)
        else:
            gnn_feat = torch.zeros(images.shape[0], self.cfg.model.gnn.output_dim,
                                   device=images.device, dtype=images.dtype)
        fused = self.fusion(cnn_feat, vit_feat, gnn_feat)
        logits = self.classifier(fused)
        out = {"logits": logits, "probs": F.softmax(logits, dim=-1)}
        if return_features:
            out["features"] = {"cnn": cnn_feat, "vit": vit_feat, "gnn": gnn_feat, "fused": fused}
        return out

    def count_parameters(self):
        def c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {k: c(v) for k,v in [("cnn",self.cnn_branch),("vit",self.vit_branch),
                ("gnn",self.gnn_branch or nn.Linear(1,1)),
                ("fusion",self.fusion),("classifier",self.classifier)]}