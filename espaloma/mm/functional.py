# =============================================================================
# IMPORTS
# =============================================================================
import math
import dgl.backend as F
import espaloma as esp

# =============================================================================
# CONSTANTS
# =============================================================================
from simtk import unit
from simtk.unit.quantity import Quantity

LJ_SWITCH = Quantity(1.0, unit.angstrom).value_in_unit(esp.units.DISTANCE_UNIT)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def linear_mixture_to_original(k1, k2, b1, b2):
    """ Translating linear mixture coefficients back to original
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
    """ Harmonic term.

    Parameters
    ----------
    x : `F.Tensor`, `shape=(batch_size, 1)`
    k : `F.Tensor`, `shape=(batch_size, len(order))`
    eq : `F.Tensor`, `shape=(batch_size, len(order))`
    order : `int` or `List` of `int`

    Returns
    -------
    u : `F.Tensor`, `shape=(batch_size, 1)`
    """

    if isinstance(order, list):
        order = F.copy_to(F.tensor(order), F.context(x))

    res = (k * ((x - eq)) ** (order[:, None, None]))

    res = F.swapaxes(res, 0, 1)
    res = F.swapaxes(res, 1, 2)
    res = F.sum(res, -1)

    return res


def periodic_fixed_phases(dihedrals, ks):
    """Periodic torsion term with n_phases = 6, periodicities = 1..n_phases, phases = zeros

    Parameters
    ----------
    dihedrals : F.Tensor, shape=(n_snapshots, n_dihedrals)
        dihedral angles -- TODO: confirm in radians?
    ks : F.Tensor, shape=(n_dihedrals, n_phases)
        force constants -- TODO: confirm in esp.unit.ENERGY_UNIT ?

    Returns
    -------
    u : F.Tensor, shape=(n_snapshots, 1)
        potential energy of each snapshot

    Notes
    -----
    TODO: is there a way to annotate / type-hint tensor shapes? (currently adding many assert statements)
    TODO: merge with esp.mm.functional.periodic -- adding this because I was having difficulty debugging runtime tensor
      shape errors in esp.mm.functional.periodic, which allows for a more flexible mix of input shapes and types
    """

    # periodicity = 1..n_phases
    n_phases = 6
    periodicity = F.arange(n_phases) + 1

    # assert input shape consistency
    n_snapshots, n_dihedrals = dihedrals.shape
    n_dihedrals_, n_phases_ = ks.shape
    assert n_dihedrals == n_dihedrals_
    assert n_phases == n_phases_

    # promote everything to this shape
    stacked_shape = (n_snapshots, n_dihedrals, n_phases)

    # duplicate ks n_snapshots times
    ks_stacked = F.stack([ks] * n_snapshots, dim=0)
    assert ks_stacked.shape == stacked_shape

    # duplicate dihedral angles n_phases times
    dihedrals_stacked = F.stack([dihedrals] * n_phases, dim=2)
    assert dihedrals_stacked.shape == stacked_shape

    # duplicate periodicity n_snapshots * n_dihedrals times
    ns = F.stack(
        [F.stack([periodicity] * n_snapshots)] * n_dihedrals, dim=1
    )
    assert ns.shape == stacked_shape

    # compute k_n * cos(n * theta) for n in 1..n_phases, for each dihedral in each snapshot
    energy_terms = ks_stacked * F.cos(ns * dihedrals_stacked)
    assert energy_terms.shape == stacked_shape

    # sum over n_dihedrals and n_phases
    energy_sums = energy_terms.sum(dim=(1, 2))
    assert energy_sums.shape == (n_snapshots,)

    return energy_sums.reshape((n_snapshots, 1))


def periodic(
    x, k, periodicity=list(range(1, 7)), phases=[0.0 for _ in range(6)]
):
    """ Periodic term.

    Parameters
    ----------
    x : `F.Tensor`, `shape=(batch_size, 1)`
    k : `F.Tensor`, `shape=(batch_size, number_of_phases)`
    periodicity: either list of length number_of_phases, or
        `F.Tensor`, `shape=(batch_size, number_of_phases)`
    phases : either list of length number_of_phases, or
        `F.Tensor`, `shape=(batch_size, number_of_phases)`
    """

    if isinstance(phases, list):
        phases = F.copy_to(F.tensor(phases), F.context(x))

    if isinstance(periodicity, list):
        periodicity = F.copy_to(F.tensor(periodicity), F.context(x))

    print(x.shape)

    if periodicity.ndim == 1:
        periodicity = F.repeat(F.repeat(periodicity[None, None, :], x.shape[0], 0), x.shape[1], 1)

    elif periodicity.ndim == 2:
        periodicity = F.repeat(periodicity[:, None, :], x.shape[1], 1)

    if phases.ndim == 1:
        phases = F.repeat(F.repeat(phases[None, None, :], x.shape[0], 0), x.shape[1], 1)

    elif phases.ndim == 2:
        phases = F.repeat(phases[:, None, :], x.shape[1], 1)

    n_theta = periodicity * x[:, :, None]

    n_theta_minus_phases = n_theta - phases

    cos_n_theta_minus_phases = F.cos(n_theta_minus_phases)

    k = F.repeat(
        k[:, None, :],
        x.shape[1],
        1
    )

    energy = (k * (1.0 + cos_n_theta_minus_phases)).sum(-1)

    return energy


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
    x, epsilon, sigma, order=[12, 6], coefficients=[1.0, 1.0], switch=LJ_SWITCH
):
    r""" Lennard-Jones term.

    Notes
    -----
    ..math::
    E  = \epsilon  ((\sigma / r) ^ {12} - (\sigma / r) ^ 6)

    Parameters
    ----------
    x : `F.Tensor`, `shape=(batch_size, 1)`
    epsilon : `F.Tensor`, `shape=(batch_size, len(order))`
    sigma : `F.Tensor`, `shape=(batch_size, len(order))`
    order : `int` or `List` of `int`
    coefficients : F.tensor or list
    switch : unitless switch width (distance)

    Returns
    -------
    u : `F.Tensor`, `shape=(batch_size, 1)`
    """
    if isinstance(order, list):
        order = F.tensor(order, device=x.device)

    if isinstance(coefficients, list):
        coefficients = F.tensor(coefficients, device=x.device)

    assert order.shape[0] == 2
    assert order.dim() == 1

    # TODO:
    # for experiments only
    # erase later

    # compute sigma over x
    sigma_over_x = sigma / x

    # erase values under switch
    sigma_over_x = F.where(
        F.lt(x, switch), F.zeros_like(sigma_over_x), sigma_over_x,
    )

    return epsilon * (
        coefficients[0] * sigma_over_x ** order[0]
        - coefficients[1] * sigma_over_x ** order[1]
    )


def gaussian(x, coefficients, phases=[idx * 0.001 for idx in range(200)]):
    r""" Gaussian basis function.

    Parameters
    ----------
    x : F.Tensor
    coefficients : list or F.Tensor of length n_phases
    phases : list or F.Tensor of length n_phases
    """

    if isinstance(phases, list):
        # (number_of_phases, )
        phases = F.tensor(phases, device=x.device)

    # broadcasting
    # (number_of_hypernodes, number_of_snapshots, number_of_phases)
    phases = phases[None, None, :].repeat(x.shape[0], x.shape[1], 1)
    x = x[:, :, None].repeat(1, 1, phases.shape[-1])
    coefficients = coefficients[:, None, :].repeat(1, x.shape[1], 1)

    return (coefficients * F.exp(-0.5 * (x - phases) ** 2)).sum(-1)


def linear_mixture(x, coefficients, phases=[0.0, 1.0]):
    r""" Linear mixture basis function.

    x : F.Tensor
    coefficients : list or F.Tensor of length 2
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

    u = u1 + u2 # - k1 * b1 ** 2 - k2 ** b2 ** 2 + b ** 2

    return u
