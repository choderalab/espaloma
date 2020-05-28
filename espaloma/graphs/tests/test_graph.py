import pytest


def test_import():
    import espaloma
    import espaloma.graphs.graph


def test_init():
    import espaloma as esp
    g = esp.Graph()
