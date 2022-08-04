import pytest
import torch


def test_graph():
    import espaloma as esp

    g = esp.Graph("c1ccccc1")

    print(g.heterograph)


@pytest.fixture
def graph():
    import espaloma as esp

    return esp.Graph("c1ccccc1")


def test_ndata_consistency(graph):
    import torch

    assert torch.equal(graph.ndata["h0"], graph.nodes["n1"].data["h0"])


@pytest.mark.parametrize("molecule, charge", [
    pytest.param("C", 0, id="methane"),
    pytest.param("[NH4+]", 1, id="Ammonium"),
    pytest.param("CC(=O)[O-]", -1, id="Acetate")
])
def test_formal_charge(molecule, charge):
    import espaloma as esp

    graph = esp.Graph(molecule)
    assert graph.nodes["g"].data["sum_q"].numpy()[0] == charge


def test_save_and_load(graph):
    import tempfile

    import espaloma as esp

    with tempfile.TemporaryDirectory() as tempdir:
        graph.save(tempdir + "/g.esp")
        new_graph = esp.Graph()
        new_graph.load(tempdir + "/g.esp")

    assert graph.homograph.number_of_nodes == graph.homograph.number_of_nodes

    assert graph.homograph.number_of_edges == graph.homograph.number_of_edges


# TODO: test offmol_indices
# TODO: test relationship_indices_from_offmol
