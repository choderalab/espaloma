# =============================================================================
# IMPORTS
# =============================================================================
import math
import torch

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
    b = (k1 * b1 + k2 * b2) / (k + 1e-3)

    return k, b


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic(x, k, eq, order=[2]):
    """ Harmonic term.

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

    return k * ((x - eq)).pow(order[:, None, None]).permute(1, 2, 0).sum(
        dim=-1
    )

def periodic(x, k, periodicity=list(range(6)), phases=[0.0 for _ in range(6)]):
    """ Periodic term.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)`
    k : `torch.Tensor`, `shape=(batch_size, number_of_phases)`
    eq: `torch.Tensor`, `shape=(batch_size, number_of_phases)`
    """

    if isinstance(phases, list):
        phases = torch.tensor(phases, device=x.device)

    if isinstance(periodicity, list):
        periodicity = torch.tensor(
            periodicity, device=x.device, dtype=torch.float32
        )

    return k * (1.0 + torch.cos(
        periodicity[None, :].repeat(x.shape[0], 1) * x
        - phases
    ))

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


def lj(x, epsilon, sigma, order=[12, 6]):
    r""" Lennard-Jones term.

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

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)`


    """
    if isinstance(order, list):
        order = torch.tensor(order, device=x.device)

    assert order.shape[0] == 2
    assert order.dim() == 1

    return epsilon * ((sigma / x) ** order[0] - (sigma / x) ** order[1])

def gaussian(x, coefficients, phases=[idx * 0.001 for idx in range(200)]):
    r""" Gaussian basis function.

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

def linear_mixture(x, coefficients, phases=[0.10, 0.25]):
    r""" Linear mixture basis function.

    """

    assert len(phases) == 2, 'Only two phases now.'
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
    k, b = linear_mixture_to_original(k1, k2, b1, b2)

    # (batch_size, 1)
    u1 = k1 * (x - b1) ** 2
    u2 = k2 * (x - b2) ** 2

    u = u1 + u2 - k1 * b1 ** 2 - k2 ** b2 ** 2 + b ** 2

    return u
