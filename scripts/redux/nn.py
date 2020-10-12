import torch
import torch.nn.functional as F
from dgl.nn.pytorch import TAGConv
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class TAG(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, k=2, activation=F.relu):
        super(TAG, self).__init__()
        self.layer1 = TAGConv(in_feats, h_feats, k, activation=activation)
        self.layer2 = TAGConv(h_feats, h_feats, k, activation=activation)
        self.layer3 = TAGConv(h_feats, num_classes, k, activation=activation)
        self.activation = activation

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = self.activation(h)
        h = self.layer2(graph, h)
        h = self.activation(h)
        h = self.layer3(graph, h)
        return h
