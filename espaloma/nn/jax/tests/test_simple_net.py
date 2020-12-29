import pytest


def test_small_net():
    import torch

    from espaloma.nn import jax as esp_nn

    layer = esp_nn.dgl_legacy.gn()
    net = esp_nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"])
