import pytest
import numpy as np
import numpy.testing as npt

def test_init():
    import espaloma as esp
    g = esp.HomogeneousGraph()

    assert g._stage == 'homogeneous'

def test_from_rdkit():
    import espaloma as esp
    
    from rdkit import Chem
    m = Chem.MolFromSmiles('c1ccccc1')

    g = esp.HomogeneousGraph(m)

    adjacency_matrix = g.adjacency_matrix()

    npt.assert_almost_equal(
            adjacency_matrix.coalesce().values().detach().numpy(),
            np.ones(12))

    assert adjacency_matrix.shape[0] == 6
    assert adjacency_matrix.shape[1] == 6
    

