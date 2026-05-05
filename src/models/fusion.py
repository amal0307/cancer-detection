import torch, torch.nn as nn, torch.nn.functional as F, math

class LearnableTemperatureAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(num_heads))
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, Lq, D = query.shape
        Lk = key.shape[1]
        Q = self.q(query).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k(key).reshape(B, Lk, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v(value).reshape(B, Lk, self.num_heads, self.head_dim).transpose(1,2)
        temp = self.temperature.abs().clamp(min=0.1).view(1, self.num_heads, 1, 1)
        attn = self.drop(F.softmax((Q @ K.transpose(-2,-1)) / (self.scale * temp), dim=-1))
        out = (attn @ V).transpose(1,2).reshape(B, Lq, D)
        return self.out(out), attn


class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=1792, vit_dim=768, gnn_dim=256,
                 fusion_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.Linear(cnn_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.vit_proj = nn.Sequential(nn.Linear(vit_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.gnn_proj = nn.Sequential(nn.Linear(gnn_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.cnn_attn_vit = LearnableTemperatureAttention(fusion_dim, num_heads, dropout)
        self.vit_attn_cnn = LearnableTemperatureAttention(fusion_dim, num_heads, dropout)
        self.gnn_attn_all = LearnableTemperatureAttention(fusion_dim, num_heads, dropout)
        ff = lambda d: nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d*4,d))
        self.ff_cnn, self.ff_vit, self.ff_gnn = ff(fusion_dim), ff(fusion_dim), ff(fusion_dim)
        self.norm_cnn = nn.LayerNorm(fusion_dim)
        self.norm_vit = nn.LayerNorm(fusion_dim)
        self.norm_gnn = nn.LayerNorm(fusion_dim)
        self.final = nn.Sequential(
            nn.Linear(fusion_dim*3, fusion_dim*2), nn.LayerNorm(fusion_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim*2, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

    def forward(self, cnn_feat, vit_feat, gnn_feat):
        cnn = self.cnn_proj(cnn_feat).unsqueeze(1)
        vit = self.vit_proj(vit_feat).unsqueeze(1)
        gnn = self.gnn_proj(gnn_feat).unsqueeze(1)
        cnn_ctx, _ = self.cnn_attn_vit(cnn, vit, vit)
        vit_ctx, _ = self.vit_attn_cnn(vit, cnn, cnn)
        gnn_ctx, _ = self.gnn_attn_all(gnn, torch.cat([cnn,vit], dim=1), torch.cat([cnn,vit], dim=1))
        cnn_o = self.norm_cnn(cnn.squeeze(1) + cnn_ctx.squeeze(1))
        vit_o = self.norm_vit(vit.squeeze(1) + vit_ctx.squeeze(1))
        gnn_o = self.norm_gnn(gnn.squeeze(1) + gnn_ctx.squeeze(1))
        cnn_o = cnn_o + self.ff_cnn(cnn_o)
        vit_o = vit_o + self.ff_vit(vit_o)
        gnn_o = gnn_o + self.ff_gnn(gnn_o)
        return self.final(torch.cat([cnn_o, vit_o, gnn_o], dim=-1))