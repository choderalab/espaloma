import numpy as np
import numpy.testing as npt
import torch


def _sample_unit_circle(n_samples: int = 1) -> np.ndarray:
    """
    >>> np.isclose(np.linalg.norm(_sample_unit_circle(1)), 1)
    True

    """
    theta = np.random.rand(n_samples) * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    xy = np.array([x, y]).T
    assert (xy.shape == (n_samples, 2))
    return xy


def _sample_four_particle_torsion_scan(n_samples: int = 1) -> np.ndarray:
    """Generate n_samples random configurations of a 4-particle system abcd where
    * distances ab, bc, cd are constant,
    * angles abc, bcd are constant
    * dihedral angle abcd is uniformly distributed in [0, 2pi]

    Returns
    -------
    xyz : np.ndarray, shape = (n_samples, 4, 3)

    Notes
    -----
    * Positions of a,b,c are constant, and x-coordinate of d is constant.
        To be more exacting, could add random displacements and rotations.
    """
    a = (-3, -1, 0)
    b = (-2, 0, 0)
    c = (-1, 0, 0)
    d = (0, 1, 0)

    # form one 3D configuration
    conf = np.array([a, b, c, d])
    assert (conf.shape == (4, 3))

    # make n_samples copies
    xyz = np.array([conf] * n_samples, dtype=float)
    assert (xyz.shape == (n_samples, 4, 3))

    # assign y and z coordinates of particle d to unit-circle samples
    xyz[:, 3, 1:] = _sample_unit_circle(n_samples)

    return xyz



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
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
        loc=torch.zeros(5, 3), scale=torch.ones(5, 3)
    )

    x0 = distribution.sample()
    x1 = distribution.sample()
    x2 = distribution.sample()
    x3 = distribution.sample()

    left = torch.cross(x1 - x0, x1 - x2, dim=-1)

    right = torch.cross(x2 - x1, x2 - x3, dim=-1)

    npt.assert_almost_equal(
        esp.mm.geometry._angle(left, right).numpy(),
        esp.dihedral(x0, x1, x2, x3).numpy(),
        decimal=3,
    )
