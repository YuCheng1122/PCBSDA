import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


class GCN_CCSA(nn.Module):
    """GCN model adapted for CCSA: returns both prediction and graph-level feature."""

    def __init__(self, num_node_features, hidden_channels, output_channels, num_classes,
                 num_layers, dropout, pooling):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(num_node_features, output_channels))
            self.bns.append(nn.BatchNorm1d(output_channels))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, output_channels))
            self.bns.append(nn.BatchNorm1d(output_channels))

        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling_type == 'mean':
            feature = global_mean_pool(x, batch)
        else:
            feature = global_add_pool(x, batch)

        feature = torch.nn.functional.normalize(feature, p=2, dim=1)

        pred = self.classifier(feature)
        return pred, feature
