# import pytest
import hgfp
import numpy as np
import numpy.testing as npt

def test_idxs_ethane():
    adjacency_matrix = np.array(
        [[0, 1, 1, 1, 1, 0, 0, 0],
         [0 ,0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)

    bond_idxs, angle_idxs, torsion_idxs, one_four_idxs, nonbonded_idxs = hgfp.geometry.idxs.get_idxs(
        adjacency_matrix)

    npt.assert_almost_equal(
        bond_idxs,
        np.array(
            [[0, 1],
             [0, 2],
             [0, 3],
             [0, 4],
             [1, 5],
             [1, 6],
             [1, 7]]))

    npt.assert_almost_equal(
        angle_idxs,
        np.array(
          [[1, 0, 2],
           [1, 0, 3],
           [2, 0, 3],
           [1, 0, 4],
           [2, 0, 4],
           [3, 0, 4],
           [0, 1, 5],
           [0, 1, 6],
           [5, 1, 6],
           [0, 1, 7],
           [5, 1, 7],
           [6, 1, 7]]))

    npt.assert_almost_equal(
        torsion_idxs,
        np.array(
        [[2, 0, 1, 5],
       [3, 0, 1, 5],
       [4, 0, 1, 5],
       [2, 0, 1, 6],
       [3, 0, 1, 6],
       [4, 0, 1, 6],
       [2, 0, 1, 7],
       [3, 0, 1, 7],
       [4, 0, 1, 7]]))

    npt.assert_almost_equal(
        one_four_idxs,
        [[2, 5],
       [3, 5],
       [4, 5],
       [2, 6],
       [3, 6],
       [4, 6],
       [2, 7],
       [3, 7],
       [4, 7]])
