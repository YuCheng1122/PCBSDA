"""
FCGAT model — GAT + global max pooling for graph-level classification.

Architecture:
  GATConv (in=100, out=192, K=3 heads, concat)  LeakyReLU
  # Set2Set (dim=192, steps=4)                   → hG ∈ R^384  (removed)
  global_max_pool                                 → hG ∈ R^192
  Dropout
  Linear  (192 → num_classes)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
# from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_max_pool


class FCGAT(nn.Module):
    def __init__(
        self,
        num_node_features: int = 100,
        hidden_channels: int = 192,
        num_classes: int = 2,
        gat_heads: int = 3,
        set2set_steps: int = 4,  # kept for API compatibility, unused
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        assert hidden_channels % gat_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by gat_heads ({gat_heads})"
        )
        head_dim = hidden_channels // gat_heads

        self.gat = GATConv(
            in_channels=num_node_features,
            out_channels=head_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
        )

        # self.set2set = Set2Set(hidden_channels, processing_steps=set2set_steps)

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # h_g = self.set2set(x, batch)
        h_g = global_max_pool(x, batch)

        return self.classifier(h_g)
