import pytest


def test_graph():
    import espaloma as esp
    g = esp.Graph('c1ccccc1')

@pytest.fixture
def graph():
    import espaloma as esp
    return esp.Graph('c1ccccc1')

def test_ndata_consistency(graph):
    import torch

    assert torch.equal(
            graph.ndata['h0'],
            graph.nodes['n1'].data['h0'])

def test_save_and_load(graph):
    import tempfile
    import espaloma as esp

    with tempfile.TemporaryDirectory() as tempdir:
        graph.save(tempdir + '/g.esp')
        new_graph = esp.Graph()
        new_graph.load(tempdir + '/g.esp')

    assert graph.homograph.number_of_nodes\
            == graph.homograph.number_of_nodes 

    assert graph.homograph.number_of_edges\
            == graph.homograph.number_of_edges


