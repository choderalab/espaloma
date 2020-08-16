# =============================================================================
# IMPORTS
# =============================================================================
import torch


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
        order = torch.tensor(order)

    return k * ((x - eq)).pow(order[:, None, None]).permute(1, 2, 0).sum(
        dim=-1
    )


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


def lj(x, epsilon, sigma, order=torch.tensor([12, 6])):
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
        order = torch.tensor(order)

    assert order.shape[0] == 2
    assert order.dim() == 1

    return epsilon * ((sigma / x) ** order[0] - (sigma / x) ** order[1])

def gaussian(x, coefficients, phases=[idx * 0.001 for idx in range(200)]):
    r""" Gaussian basis function.

    """
    if isinstance(phases, list):
        # (number_of_phases, )
        phases = torch.tensor(phases)

    # broadcasting
    # (number_of_hypernodes, number_of_snapshots, number_of_phases)
    phases = phases[None, None, :].repeat(x.shape[0], x.shape[1], 1)
    x = x[:, :, None].repeat(1, 1, phases.shape[-1])
    coefficients = coefficients[:, None, :].repeat(1, x.shape[1], 1)


    return (coefficients * torch.exp(-0.5 * (x - phases) ** 2)).sum(-1)
