import torch

torch.set_default_dtype(torch.float64)

from espaloma.units import DISTANCE_UNIT, ENERGY_UNIT

from simtk import unit

distance_to_nm = (1.0 * DISTANCE_UNIT).value_in_unit(unit.nanometer)
energy_from_kjmol = (1.0 * unit.kilojoule_per_mole).value_in_unit(ENERGY_UNIT)


def step(x):
    """return (x > 0)"""
    return 1.0 * (x >= 0)


def _gbsa_obc2_energy_omm(
        distance_matrix,
        radii, scales, charges,
        alpha=0.8, beta=0.0, gamma=2.909125,
        dielectric_offset=0.009,
        surface_tension=28.3919551,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
        probe_radius=0.14
):
    """
    Assume everything is given in OpenMM units
    ported from jax/numpy implementation here:
    https://github.com/openforcefield/bayes-implicit-solvent/blob/067239fcbb8af28eb6310d702804887662692ec2/bayes_implicit_solvent/gb_models/jax_gb_models.py#L13-L60

    with corrections and refinements by Yutong Zhao here
    https://github.com/proteneer/timemachine/blob/417f4b0b1181b638935518532c78c380b03d7d19/timemachine/potentials/gbsa.py#L1-L111
    """

    N = len(charges)
    eye = torch.eye(N, dtype=distance_matrix.dtype)
    assert (eye.shape == (N, N))

    r = distance_matrix + eye
    assert (r.shape == (N, N))
    or1 = radii.reshape((N, 1)) - dielectric_offset
    or2 = radii.reshape((1, N)) - dielectric_offset

    assert (or1.shape == (N, 1))
    assert (or2.shape == (1, N))

    sr2 = scales.reshape((1, N)) * or2
    assert (sr2.shape == (1, N))

    r_sr2 = abs(r - sr2)
    assert (r_sr2.shape == (N, N))

    L = torch.max(or1, r_sr2)
    # TODO: check if elementwise
    #print('L.shape', L.shape)
    #print('or1.shape', or1.shape)
    #print('r_sr2.shape', r_sr2.shape)
    # TODO: this is fishy
    # assert(L.shape == or1.shape)

    U = r + sr2
    assert (U.shape == (N, N))

    I = 1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (
            1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * torch.log(
        L / U) / r
    assert (I.shape == (N, N))

    # handle the interior case
    condition = or1 < (sr2 - r)
    if_true = I + 2 * (1 / or1 - 1 / L)
    if_false = I
    I = torch.where(condition, if_true, if_false)

    I = step(r + sr2 - or1) * 0.5 * I  # note the extra 0.5 here

    # intention: zero out values on the diagonal
    # TODO: possibly replace this diag(diag) with scatter update
    I -= torch.diag(torch.diag(I))

    I = torch.sum(I, dim=1)

    # okay, next compute born radii
    offset_radius = radii - dielectric_offset

    psi = I * offset_radius

    psi_coefficient = alpha
    psi2_coefficient = beta
    psi3_coefficient = gamma

    psi_term = (psi_coefficient * psi) - (psi2_coefficient * psi ** 2) + (
            psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - torch.tanh(psi_term) / radii)
    assert (B.shape == (N,))

    E = 0.0
    # single particle
    # ACE
    ACE_individual_terms = surface_tension * (radii + probe_radius) ** 2 * (
                radii / B) ** 6
    ACE = torch.sum(ACE_individual_terms)
    assert (ACE_individual_terms.shape == (N,))
    E += ACE

    # on-diagonal
    on_diagonal = -0.5 * (
            1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B
    assert (on_diagonal.shape == (N,))
    E += torch.sum(on_diagonal)

    # particle pair
    # note: np.outer --> torch.ger
    assert (B.ndim == 1)

    f = torch.sqrt(r ** 2 + torch.ger(B, B) * torch.exp(
        -r ** 2 / (4 * torch.ger(B, B))))
    charge_products = torch.ger(charges, charges)
    assert (f.shape == (N, N))
    assert (charge_products.shape == (N, N))

    ixns = - (
            1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f

    E += torch.sum(torch.triu(ixns, diagonal=1))
    return E  # E is in kJ/mol at this point


def gbsa_obc2_energy(
        distance_matrix_in_bohr,
        radii_in_bohr, scales, charges,
        alpha=0.8, beta=0.0, gamma=2.909125,
        dielectric_offset=0.009,
        surface_tension=28.3919551,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
        probe_radius=0.14
):
    # convert distances and radii into units of nanometers before proceeding
    distance_matrix = distance_matrix_in_bohr * distance_to_nm
    radii = radii_in_bohr * distance_to_nm

    E = _gbsa_obc2_energy_omm(
        distance_matrix,
        radii, scales, charges,
        alpha, beta, gamma,
        dielectric_offset=dielectric_offset,
        surface_tension=surface_tension,
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        probe_radius=probe_radius,
    )


    return E * energy_from_kjmol  # return E in espaloma energy unit
