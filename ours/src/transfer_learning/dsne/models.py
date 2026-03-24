import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, AttentionalAggregation, global_add_pool, global_mean_pool


class GCN_DSNE(nn.Module):
    """GCN model adapted for d-SNE: returns both prediction and graph-level feature."""

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

        if pooling == 'attention':
            self.att_gate = nn.Sequential(
                nn.Linear(output_channels, output_channels // 2),
                nn.ReLU(),
                nn.Linear(output_channels // 2, 1)
            )
            self.pool = AttentionalAggregation(gate_nn=self.att_gate)

        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling_type == 'mean':
            feature = global_mean_pool(x, batch)
        elif self.pooling_type == 'attention':
            feature = self.pool(x, batch)
        else:
            feature = global_add_pool(x, batch)

        feature = F.normalize(feature, p=2, dim=1)

        pred = self.classifier(feature)
        return pred, feature


class GAT_DSNE(nn.Module):
    """GAT model adapted for d-SNE: returns both prediction and graph-level feature."""

    def __init__(self, num_node_features, hidden_channels, output_channels, num_classes,
                 num_layers, dropout, pooling, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(num_node_features, output_channels,
                                      heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(output_channels))
        else:
            self.convs.append(GATConv(num_node_features, hidden_channels,
                                      heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                          heads=heads, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            self.convs.append(GATConv(hidden_channels * heads, output_channels,
                                      heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(output_channels))

        if pooling == 'attention':
            self.att_gate = nn.Sequential(
                nn.Linear(output_channels, output_channels // 2),
                nn.ReLU(),
                nn.Linear(output_channels // 2, 1)
            )
            self.pool = AttentionalAggregation(gate_nn=self.att_gate)

        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling_type == 'mean':
            feature = global_mean_pool(x, batch)
        elif self.pooling_type == 'attention':
            feature = self.pool(x, batch)
        else:
            feature = global_add_pool(x, batch)

        feature = F.normalize(feature, p=2, dim=1)

        pred = self.classifier(feature)
        return pred, feature
