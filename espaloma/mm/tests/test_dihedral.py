import numpy.testing as npt
import torch

import espaloma as esp
from espaloma.utils.geometry import (
    _sample_four_particle_torsion_scan,
    _timemachine_signed_torsion_angle,
)


def test_dihedral_vectors():
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
        loc=torch.zeros(5, 3), scale=torch.ones(5, 3)
    )

    left = distribution.sample()
    right = distribution.sample()

    npt.assert_almost_equal(
        esp.mm.geometry._angle(left, right).numpy(),
        esp.mm.geometry._dihedral(left, right).numpy(),
        decimal=3,
    )


def test_dihedral_points():
    n_samples = 1000

    # get geometries
    xyz_np = _sample_four_particle_torsion_scan(n_samples)

    # compute dihedrals using timemachine (numpy / JAX)
    ci, cj, ck, cl = (
        xyz_np[:, 0, :],
        xyz_np[:, 1, :],
        xyz_np[:, 2, :],
        xyz_np[:, 3, :],
    )
    theta_timemachine = _timemachine_signed_torsion_angle(ci, cj, ck, cl)

    # compute dihedrals using espaloma (PyTorch)
    xyz = torch.tensor(xyz_np)
    x0, x1, x2, x3 = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :], xyz[:, 3, :]
    theta_espaloma = esp.dihedral(x0, x1, x2, x3).numpy()

    npt.assert_almost_equal(
        theta_timemachine,
        theta_espaloma,
        decimal=3,
    )
