

import jax.numpy as np


def step(x):
    # return (x > 0)
    return 1.0 * (x >= 0)


def compute_OBC_energy_bayes_implicit(distance_matrix, radii, scales, charges,
                                  offset=0.009, screening=138.935484,
                                  surface_tension=28.3919551,
                                  solvent_dielectric=78.5,
                                  solute_dielectric=1.0,
                                  ):
    """From https://github.com/openforcefield/bayes-implicit-solvent/blob/46936da65ed93ed33f0f97362a1dea12f9170758/bayes_implicit_solvent/gb_models/jax_gb_models.py

    in turn based on https://github.com/openmm/openmm/blob/master/platforms/reference/src/SimTKReference/ReferenceObc.cpp
    """
    N = len(radii)
    # print(type(distance_matrix))
    eye = np.eye(N, dtype=distance_matrix.dtype)
    # print(type(eye))
    r = distance_matrix + eye  # so I don't have divide-by-zero nonsense
    or1 = radii.reshape((N, 1)) - offset
    or2 = radii.reshape((1, N)) - offset
    sr2 = scales.reshape((1, N)) * or2

    L = np.maximum(or1, abs(r - sr2))
    U = r + sr2
    I = step(r + sr2 - or1) * 0.5 * (
            1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (
                1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * np.log(
        L / U) / r)

    I -= np.diag(np.diag(I))
    I = np.sum(I, axis=1)

    # okay, next compute born radii
    offset_radius = radii - offset
    psi = I * offset_radius
    psi_coefficient = 0.8
    psi2_coefficient = 0
    psi3_coefficient = 2.909125

    psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (
                psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - np.tanh(psi_term) / radii)

    # finally, compute the three energy terms
    E = 0.0

    # single particle
    E += np.sum(surface_tension * (radii + 0.14) ** 2 * (radii / B) ** 6)
    E += np.sum(-0.5 * screening * (
                1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B)

    # particle pair
    f = np.sqrt(
        r ** 2 + np.outer(B, B) * np.exp(-r ** 2 / (4 * np.outer(B, B))))
    charge_products = np.outer(charges, charges)

    E += np.sum(np.triu(-screening * (
                1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f,
                        k=1))

    return E
