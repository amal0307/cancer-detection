import torch, torch.nn as nn, timm

class ViTBranch(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224", pretrained=True,
                 embedding_dim=768, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        vit_dim = self.backbone.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, embedding_dim),
            nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.embedding_dim = embedding_dim

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def forward(self, x):
        return self.proj(self.backbone(x))