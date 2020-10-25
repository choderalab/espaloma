import torch

torch.set_default_dtype(torch.float64)

from espaloma.units import DISTANCE_UNIT, ENERGY_UNIT

from simtk import unit

distance_to_nm = (1.0 * DISTANCE_UNIT).value_in_unit(unit.nanometer)
energy_from_kjmol = (1.0 * unit.kilojoule_per_mole).value_in_unit(ENERGY_UNIT)


def step(x):
    """return (x > 0)"""
    return 1.0 * (x >= 0)


def gbsa_obc2_energy(
        distance_matrix: torch.tensor,
        radii: torch.tensor,
        scales: torch.tensor,
        charges: torch.tensor,
        offset=0.009, screening=138.935484,
        surface_tension=28.3919551,
        solvent_dielectric=78.5, solute_dielectric=1.0
):
    """
    ported from jax/numpy implementation here:
    https://github.com/openforcefield/bayes-implicit-solvent/blob/067239fcbb8af28eb6310d702804887662692ec2/bayes_implicit_solvent/gb_models/jax_gb_models.py#L13-L60
    """
    # convert distances and radii into units of nanometers before proceeding
    distance_matrix_in_nm = distance_matrix * distance_to_nm
    radii_in_nm = radii * distance_to_nm

    # scales are unitless

    N = len(radii_in_nm)
    eye = torch.eye(N, dtype=distance_matrix_in_nm.dtype)
    r = distance_matrix_in_nm + eye
    or1 = radii_in_nm.reshape((N, 1)) - offset
    or2 = radii_in_nm.reshape((1, N)) - offset
    sr2 = scales.reshape((1, N)) * or2

    L = torch.max(or1, torch.abs(r - sr2))
    U = r + sr2

    # https://github.com/openforcefield/bayes-implicit-solvent/blob/067239fcbb8af28eb6310d702804887662692ec2/bayes_implicit_solvent/gb_models/jax_gb_models.py#L29-L31
    #  but incorporating bugfix Yutong pointed out
    I = 1 / L \
        - 1 / U \
        + 0.25 * (r - sr2 ** 2 / r) * (1 / (U ** 2) - 1 / (L ** 2)) \
        + 0.5 * torch.log(L / U) / r
    I = torch.where(or1 < (sr2 - r), I + 2 * (1 / or1 - 1 / L), I)
    I = step(r + sr2 - or1) * 0.5 * I

    I -= torch.diag(torch.diag(I))
    I = torch.sum(I, dim=1)

    # okay, next compute born radii
    offset_radius = radii_in_nm - offset
    psi = I * offset_radius
    psi_coefficient = 0.8
    psi2_coefficient = 0
    psi3_coefficient = 2.909125

    psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (
            psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - torch.tanh(psi_term) / radii_in_nm)

    # finally, compute the three energy terms
    E = 0.0

    # single particle
    E += torch.sum(surface_tension * (radii_in_nm + 0.14) ** 2 * (radii_in_nm / B) ** 6)
    E += torch.sum(-0.5 * screening * (
            1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B)

    # particle pair
    # note: np.outer --> torch.ger
    f = torch.sqrt(
        r ** 2 + torch.ger(B, B) * torch.exp(-r ** 2 / (4 * torch.ger(B, B))))
    charge_products = torch.ger(charges, charges)

    E += torch.sum(
        torch.triu(
            -screening * (
                    + 1 / solute_dielectric
                    - 1 / solvent_dielectric
            ) * charge_products / f,
            diagonal=1
        )
    )

    # E is in kJ/mol at this point
    # return E in espaloma energy unit
    return E * energy_from_kjmol
