# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic(x, k, eq, order=2):
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

    return torch.sum(k * (x - eq) ** order, dim=-1, keepdim=True)

def periodic(x, k, eq, order):
    """ Periodic term.

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

    return torch.sum(
        k * (1.0 + torch.cos(order * x - eq)), dim=-1, keepdim=True
    )


def lj(x, k, eq, order=torch.tensor([12, 6])):
    r""" Lennard-Jones term.

    Notes
    -----
    ..math::
    E  = \epsilon  ((\sigma / r) ^ {12} - (\sigma / r) ^ 6)

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

    assert order.shape[0] == 2
    assert order.dim() == 1

    return k * ((eq / x) ** order[0] - (eq / x) ** order[1])
