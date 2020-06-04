import pytest

def test_small_net():
    import espaloma as esp
    import torch

    layer = esp.nn.dgl_legacy.gn()
    net = esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"])


