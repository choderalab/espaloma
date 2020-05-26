import pytest
import numpy as np
import numpy.testing as npt
import torch

def test_dihedral_vectors():
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
            loc=torch.zeros(5, 3),
            scale=torch.ones(5, 3))

    left = distribution.sample()
    right = distribution.sample()


    npt.assert_almost_equal(
            esp.mm.geometry._angle(left, right).numpy(),
            esp.mm.geometry._dihedral(left, right).numpy())



def test_dihedral_points():
    import espaloma as esp
    distribution = torch.distributions.normal.Normal(
            loc=torch.zeros(5, 3),
            scale=torch.ones(5, 3))

    x0 = distribution.sample()
    x1 = distribution.sample()
    x2 = distribution.sample()
    x3 = distribution.sample()

    left = torch.cross(
            x1 - x0,
            x1 - x2,
            dim=-1)

    right = torch.cross(
            x2 - x1,
            x2 - x3,
            dim=-1)

    npt.assert_almost_equal(
            esp.mm.geometry._angle(left, right).numpy(),
            esp.dihedral(x0, x1, x2, x3).numpy())


