import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax


class GEMAL(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 dropout=0.5, embed_dim=300):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.att_weight = nn.Linear(hidden_channels, 1)
        self.graph_proj = nn.Linear(hidden_channels, embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        att_score = F.leaky_relu(self.att_weight(x))
        att_score = softmax(att_score, batch)

        g = global_add_pool(att_score * x, batch)
        g = self.graph_proj(g)

        return self.classifier(g)
