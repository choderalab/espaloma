import pytest

def test_idxs_idxs():
    import espaloma as esp
    
    from rdkit import Chem
    m = Chem.MolFromSmiles('c1ccccc1')

    g = esp.HomogeneousGraph(m)

    adjacency_matrix = g.adjacency_matrix()

    idxs = esp.graphs.utils.read_heterogeneous_graph\
        .relationship_indices_from_adjacency_matrix(adjacency_matrix)

    # TODO:
    # more tests