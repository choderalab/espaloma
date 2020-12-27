import pytest


def test_small_net():
    import torch

    import espaloma as esp

    layer = esp.nn.dgl_legacy.gn()
    net = esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"])
