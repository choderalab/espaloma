import numpy as np


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


def _timemachine_signed_torsion_angle(ci, cj, ck, cl):
    """Reference implementation from Yutong Zhao's timemachine

    Copied directly from
    https://github.com/proteneer/timemachine/blob/1a0ab45e605dc1e28c44ea90f38cb0dedce5c4db/timemachine/potentials/bonded.py#L152-L199
    (but with 3 lines of dead code removed, and delta_r inlined)
    """

    rij = cj - ci
    rkj = cj - ck
    rkl = cl - ck

    n1 = np.cross(rij, rkj)
    n2 = np.cross(rkj, rkl)

    y = np.sum(np.multiply(np.cross(n1, n2), rkj / np.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
    x = np.sum(np.multiply(n1, n2), -1)

    return np.arctan2(y, x)