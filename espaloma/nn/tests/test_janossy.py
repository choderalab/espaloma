import pytest


def test_small_net():
    import torch

    import espaloma as esp

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn("GraphConv")

    # define a representation
    representation = esp.nn.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )

    # define a readout
    readout = esp.nn.readout.janossy.JanossyPooling(
        config=[32, "tanh"], in_features=32
    )

    net = torch.nn.Sequential(representation, readout)

    g = esp.Graph("c1ccccc1")
