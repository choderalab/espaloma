import pytest

def test_import():
    import espaloma
    import espaloma.graphs.graph

def test_init():
    import espaloma as esp
    
    with pytest.raises(TypeError):
        g = esp.Graph()
