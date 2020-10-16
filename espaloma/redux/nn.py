from typing import Callable

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import TAGConv
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: Callable = F.relu):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class TAG(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, k: int = 2, activation: Callable = F.relu):
        super(TAG, self).__init__()
        self.layer1 = TAGConv(in_dim, hidden_dim, k, activation=activation)
        self.layer2 = TAGConv(hidden_dim, hidden_dim, k, activation=activation)
        self.layer3 = TAGConv(hidden_dim, out_dim, k, activation=activation)
        self.activation = activation

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = self.activation(h)
        h = self.layer2(graph, h)
        h = self.activation(h)
        h = self.layer3(graph, h)
        return h
