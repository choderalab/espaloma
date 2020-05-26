import pytest

def test_small_net():
    import espaloma as esp
    ds = esp.data.esol()[:16]
    ds = esp.data.utils.batch(ds, 8)

    layer = esp.nn.dgl_legacy.gn()
    net = esp.nn.Sequential(layer, [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    net(ds[0][0])

