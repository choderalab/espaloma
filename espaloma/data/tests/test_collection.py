import pytest

def test_esol():
    import espaloma as esp
    esol = esp.data.ESOL()
    next(iter(esol))
