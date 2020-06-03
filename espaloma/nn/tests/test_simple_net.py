import pytest

def test_small_net():
    import espaloma as esp
    import torch

    ds = torch.utils.data.DataLoader(
            esp.data.ESOL()[:16].to_homogeneous_with_legacy_typing(),
            collate_fn=esp.data.utils.collate_fn)

    layer = esp.nn.dgl_legacy.gn()
    net = esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"])

    g = next(iter(ds))

    print(net(g))
