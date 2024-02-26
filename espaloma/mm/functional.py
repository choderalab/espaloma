# =============================================================================
# IMPORTS
# =============================================================================
import math

import espaloma as esp
import torch
# =============================================================================
# CONSTANTS
# =============================================================================
from openmm import unit
from openmm.unit import Quantity

LJ_SWITCH = Quantity(1.0, unit.angstrom).value_in_unit(
    esp.units.DISTANCE_UNIT
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def linear_mixture_to_original(k1, k2, b1, b2):
    """Translating linear mixture coefficients back to original
    parameterization.
    """
    # (batch_size, )
    k = k1 + k2

    # (batch_size, )
    b = (k1 * b1 + k2 * b2) / (k + 1e-7)

    return k, b


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic(x, k, eq, order=[2]):
    """Harmonic term.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)`
    k : `torch.Tensor`, `shape=(batch_size, len(order))`
    eq : `torch.Tensor`, `shape=(batch_size, len(order))`
    order : `int` or `List` of `int`

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)`
    """

    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    return (
        0.5
        * k
        * ((x - eq)).pow(order[:, None, None]).permute(1, 2, 0).sum(dim=-1)
    )

def cubic_expansion(x, k, eq, order=[2]):
    """
    Cubic expansion, eq (3) from Merck94
    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    
    delta = ((x - eq))
    delta_squared =((x - eq)).pow(order[:, None, None])
    cb = -0.007
    
    out = k * delta_squared / 2 * (1 + cb * delta)
    return out.permute(1, 2, 0).sum(dim=-1)


def oop_expansion(x, k, order=[2]):
    """
    Cubic expansion, eq (3) from Merck94
    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)


    delta_squared = x.pow(order[:, None, None])
    
    out = k * delta_squared / 2
    return out.permute(1, 2, 0).sum(dim=-1)




def near_linear_expansion(x, k, eq, order=[2]):
    """
    Near-linear angles from eq (4) from Merck94. It's basically constant
    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    theta_cos = x.cos() # eq (4)
    
    out = k * (1 + theta_cos)
    return out


def quartic_expansion(x, k, eq, order=[2]):
    """
    Eq (2) MMFF94

    Delta_r = x - eq
    k force constant md/A
    cs A

    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    
    delta = ((x - eq))
    delta_squared =((x - eq)).pow(order[:, None, None])
    cs = -2
    
    out = k * delta_squared / 2 * (1 + cs * delta + 7/12 * cs**2 * delta_squared)
    # 1 x N_atoms x 50
    return out.permute(1, 2, 0).sum(dim=-1)


def stretch_bend_expansion(x, k, eq, eq_ij, eq_kj, x_ij, x_kj, is_linear):
    """
    Eq (5) MMFF. Note that it's not applied when angle is near-linear
    """

    
    delta_ij = ((x_ij - eq_ij))
    delta_kj = ((x_kj - eq_kj))

    delta_ijk = ((x - eq))
    k_ijk = k[:, 0][:, None]
    k_kji = k[:, 1][:, None]

    
    # Condition for eq4 to apply
    #is_linear = torch.all((delta_ijk < torch.pi) * (delta_ijk > torch.pi/2), 1)[:, None].repeat(1, delta_ijk.shape[1])

    out = ((k_ijk * delta_ij + k_kji * delta_kj) * delta_ijk) #.permute(1, 2, 0).sum(dim=-1)
    
    return torch.where(is_linear, torch.zeros_like(out), out)

def periodic_fixed_phases(
    dihedrals: torch.Tensor, ks: torch.Tensor
) -> torch.Tensor:
    """Periodic torsion term with n_phases = 6, periodicities = 1..n_phases, phases = zeros

    Parameters
    ----------
    dihedrals : torch.Tensor, shape=(n_snapshots, n_dihedrals)
        dihedral angles -- TODO: confirm in radians?
    ks : torch.Tensor, shape=(n_dihedrals, n_phases)
        force constants -- TODO: confirm in esp.unit.ENERGY_UNIT ?

    Returns
    -------
    u : torch.Tensor, shape=(n_snapshots, 1)
        potential energy of each snapshot

    Notes
    -----
    TODO: is there a way to annotate / type-hint tensor shapes? (currently adding many assert statements)
    TODO: merge with esp.mm.functional.periodic -- adding this because I was having difficulty debugging runtime tensor
      shape errors in esp.mm.functional.periodic, which allows for a more flexible mix of input shapes and types
    """

    # periodicity = 1..n_phases
    n_phases = 6
    periodicity = torch.arange(n_phases) + 1

    # assert input shape consistency
    n_snapshots, n_dihedrals = dihedrals.shape
    n_dihedrals_, n_phases_ = ks.shape
    assert n_dihedrals == n_dihedrals_
    assert n_phases == n_phases_

    # promote everything to this shape
    stacked_shape = (n_snapshots, n_dihedrals, n_phases)

    # duplicate ks n_snapshots times
    ks_stacked = torch.stack([ks] * n_snapshots, dim=0)
    assert ks_stacked.shape == stacked_shape

    # duplicate dihedral angles n_phases times
    dihedrals_stacked = torch.stack([dihedrals] * n_phases, dim=2)
    assert dihedrals_stacked.shape == stacked_shape

    # duplicate periodicity n_snapshots * n_dihedrals times
    ns = torch.stack(
        [torch.stack([periodicity] * n_snapshots)] * n_dihedrals, dim=1
    )
    assert ns.shape == stacked_shape

    # compute k_n * cos(n * theta) for n in 1..n_phases, for each dihedral in each snapshot
    energy_terms = ks_stacked * torch.cos(ns * dihedrals_stacked)
    assert energy_terms.shape == stacked_shape

    # sum over n_dihedrals and n_phases
    energy_sums = energy_terms.sum(dim=(1, 2))
    assert energy_sums.shape == (n_snapshots,)

    return energy_sums.reshape((n_snapshots, 1))


def periodic(
    x, k, periodicity=list(range(1, 7)), phases=[0.0 for _ in range(6)]
):
    """Periodic term.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)`
    k : `torch.Tensor`, `shape=(batch_size, number_of_phases)`
    periodicity: either list of length number_of_phases, or
        `torch.Tensor`, `shape=(batch_size, number_of_phases)`
    phases : either list of length number_of_phases, or
        `torch.Tensor`, `shape=(batch_size, number_of_phases)`
    """

    if isinstance(phases, list):
        phases = torch.tensor(phases, device=x.device)

    if isinstance(periodicity, list):
        periodicity = torch.tensor(
            periodicity,
            device=x.device,
            dtype=torch.get_default_dtype(),
        )
    
    if periodicity.ndim == 1:
        periodicity = periodicity[None, None, :].repeat(
            x.shape[0], x.shape[1], 1
        )

    elif periodicity.ndim == 2:
        periodicity = periodicity[:, None, :].repeat(1, x.shape[1], 1)

    if phases.ndim == 1:
        phases = phases[None, None, :].repeat(
            x.shape[0],
            x.shape[1],
            1,
        )

    elif phases.ndim == 2:
        phases = phases[:, None, :].repeat(
            1,
            x.shape[1],
            1,
        )

    n_theta = periodicity * x[:, :, None]

    n_theta_minus_phases = n_theta - phases

    cos_n_theta_minus_phases = n_theta_minus_phases.cos()

    k = k[:, None, :].repeat(1, x.shape[1], 1)

    # energy = (k * (1.0 + cos_n_theta_minus_phases)).sum(dim=-1)

    energy = (
        torch.nn.functional.relu(k) * (cos_n_theta_minus_phases + 1.0)
        - torch.nn.functional.relu(0.0 - k) * (cos_n_theta_minus_phases - 1.0)
    ).sum(dim=-1)

    return energy

def periodic_mmff(
    x, k, periodicity=list(range(1, 4))
):
    """
    cos_n_theta_minus_phases multiplied by constants
    MMFF94
    """
    if isinstance(periodicity, list):
        periodicity = torch.tensor(
            periodicity,
            device=x.device,
            dtype=torch.get_default_dtype(),
        )
    
    if periodicity.ndim == 1:
        periodicity = periodicity[None, None, :].repeat(
            x.shape[0], x.shape[1], 1
        )

    elif periodicity.ndim == 2:
        periodicity = periodicity[:, None, :].repeat(1, x.shape[1], 1)


    n_theta = periodicity * x[:, :, None]
    
    cos_n_theta = n_theta.cos()
    k = torch.nn.functional.relu(k[:, None, :].repeat(1, 1, x.shape[1]))

    return k[:, :, 0] * (1 + cos_n_theta[:, :, 0]) + k[:, :, 1] * (1 - cos_n_theta[:, :, 1]) + k[:, :, 2] * (1 + cos_n_theta[:, :, 2])


# simple implementation
# def harmonic(x, k, eq):
#     return k * (x - eq) ** 2
#
# def harmonic_re(x, k, eq, a=0.0, b=0.3):
#     # temporary
#     ka = k
#     kb = eq
#
#     c = ((ka * a + kb * b) / (ka + kb)) ** 2 - a ** 2 - b ** 2
#
#     return ka * (x - a) ** 2 + kb * (x - b) ** 2

def lj(
    x,
    epsilon,
    sigma,
    order=[12, 6],
    coefficients=[1.0, 1.0],
    switch=LJ_SWITCH,
):
    r"""Lennard-Jones term.

    Notes
    -----
    ..math::
    E  = \epsilon  ((\sigma / r) ^ {12} - (\sigma / r) ^ 6)

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)`
    epsilon : `torch.Tensor`, `shape=(batch_size, len(order))`
    sigma : `torch.Tensor`, `shape=(batch_size, len(order))`
    order : `int` or `List` of `int`
    coefficients : torch.tensor or list
    switch : unitless switch width (distance)

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)`
    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    if isinstance(coefficients, list):
        coefficients = torch.tensor(coefficients, device=x.device)

    assert order.shape[0] == 2
    assert order.dim() == 1

    # TODO:
    # for experiments only
    # erase later

    # compute sigma over x
    sigma_over_x = sigma / x

    # erase values under switch
    sigma_over_x = torch.where(
        torch.lt(x, switch),
        torch.zeros_like(sigma_over_x),
        sigma_over_x,
    )

    return epsilon * (
        coefficients[0] * sigma_over_x ** order[0]
        - coefficients[1] * sigma_over_x ** order[1]
    )


def gaussian(x, coefficients, phases=[idx * 0.001 for idx in range(200)]):
    r"""Gaussian basis function.

    Parameters
    ----------
    x : torch.Tensor
    coefficients : list or torch.Tensor of length n_phases
    phases : list or torch.Tensor of length n_phases
    """

    if isinstance(phases, list):
        # (number_of_phases, )
        phases = torch.tensor(phases, device=x.device)

    # broadcasting
    # (number_of_hypernodes, number_of_snapshots, number_of_phases)
    phases = phases[None, None, :].repeat(x.shape[0], x.shape[1], 1)
    x = x[:, :, None].repeat(1, 1, phases.shape[-1])
    coefficients = coefficients[:, None, :].repeat(1, x.shape[1], 1)

    return (coefficients * torch.exp(-0.5 * (x - phases) ** 2)).sum(-1)


def linear_mixture(x, coefficients, phases=[0.0, 1.0]):
    r"""Linear mixture basis function.

    x : torch.Tensor
    coefficients : list or torch.Tensor of length 2
    phases : list of length 2
    """

    assert len(phases) == 2, "Only two phases now."
    assert coefficients.shape[-1] == 2

    # partition the dimensions
    # (, )
    b1 = phases[0]
    b2 = phases[1]

    # (batch_size, 1)
    k1 = coefficients[:, 0][:, None]
    k2 = coefficients[:, 1][:, None]

    # get the original parameters
    # (batch_size, )
    # k, b = linear_mixture_to_original(k1, k2, b1, b2)

    # (batch_size, 1)
    u1 = k1 * (x - b1) ** 2
    u2 = k2 * (x - b2) ** 2

    u = 0.5 * (u1 + u2)  # - k1 * b1 ** 2 - k2 ** b2 ** 2 + b ** 2

    return u


def harmonic_periodic_coupled(
    x_harmonic,
    x_periodic,
    k,
    eq,
    periodicity=list(range(1, 3)),
):

    if isinstance(periodicity, list):
        periodicity = torch.tensor(
            periodicity,
            device=x_harmonic.device,
            dtype=torch.get_default_dtype(),
        )

    n_theta = (
        periodicity[None, None, :].repeat(
            x_periodic.shape[0], x_periodic.shape[1], 1
        )
        * x_periodic[:, :, None]
    )

    cos_n_theta = n_theta.cos()

    k = k[:, None, :].repeat(1, x_periodic.shape[1], 1)

    sum_k_cos_n_theta = (k * cos_n_theta).sum(dim=-1)

    x_minus_eq = x_harmonic - eq

    energy = x_minus_eq * sum_k_cos_n_theta

    return energy


def harmonic_harmonic_coupled(
    x0,
    x1,
    eq0,
    eq1,
    k,
):
    energy = k * (x0 - eq0) * (x1 - eq1)
    return energy


def harmonic_harmonic_periodic_coupled(
    theta0,
    theta1,
    eq0,
    eq1,
    phi,
    k,
):
    energy = k * (theta0 - eq0) * (theta1 - eq1) * phi.cos()
    return energy
