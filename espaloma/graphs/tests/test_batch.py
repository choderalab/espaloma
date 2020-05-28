import pytest

def test_batch_unbatch():
    import espaloma as esp

    from rdkit import Chem

    m = Chem.MolFromSmiles("c1ccccc1")

    g = esp.HomogeneousGraph(m)

    hg = esp.HeterogeneousGraph(g)

    hgs = esp.graphs.heterogeneous_graph.batch([hg, hg])

    hg_new, _ = esp.graphs.heterogeneous_graph.unbatch(hgs)

    assert isinstance(hg_new, esp.HeterogeneousGraph)
