import torch, torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_dim//heads, heads=heads,
                              dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        return self.drop(self.act(self.norm(self.conv(x, edge_index) + self.res(x))))


class GNNBranch(nn.Module):
    def __init__(self, num_node_features=512, hidden_dim=256,
                 output_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.input_proj = nn.Sequential(nn.Linear(num_node_features, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())
        self.gat_layers = nn.ModuleList([
            GATBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, graph_batch):
        x = self.input_proj(graph_batch.x)
        for layer in self.gat_layers:
            x = layer(x, graph_batch.edge_index)
        return self.readout(torch.cat([
            global_mean_pool(x, graph_batch.batch),
            global_max_pool(x, graph_batch.batch)], dim=-1))