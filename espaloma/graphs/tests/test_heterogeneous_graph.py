import pytest


def test_import():
    import espaloma as esp
    import espaloma.graphs.heterogeneous_graph


def test_from_homogeneous():
    import espaloma as esp

    from rdkit import Chem

    m = Chem.MolFromSmiles("c1ccccc1")

    g = esp.HomogeneousGraph(m)

    hg = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous(g)
