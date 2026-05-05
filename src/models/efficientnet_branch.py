import torch, torch.nn as nn, timm

class EfficientNetBranch(nn.Module):
    def __init__(self, backbone="efficientnet_b4", pretrained=True,
                 embedding_dim=1792, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_dim),
            nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self._intermediate_features = None
        list(self.backbone.children())[-2].register_forward_hook(
            lambda m, i, o: setattr(self, '_intermediate_features', o.detach()))

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def forward(self, x):
        return self.proj(self.backbone(x))