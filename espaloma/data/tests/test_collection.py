import pytest


@pytest.fixture
def esol():
    import espaloma as esp

    return esp.data.esol(first=16)


def test_view(esol):
    view = esol.view(batch_size=4)
    import dgl

    graphs = list(view)
    assert len(graphs) == 4
    assert all(isinstance(graph, dgl.DGLHeteroGraph) for graph in graphs)


def test_typing(esol):
    import espaloma as esp

    typing = esp.graphs.legacy_force_field.LegacyForceField("gaff-1.81")
    esol = esol.apply(typing, in_place=True)
    view = esol.view(batch_size=4)
    for g in view:
        assert g.nodes["n1"].data["legacy_typing"].shape[0] == g.number_of_nodes(
            ntype="n1"
        )
