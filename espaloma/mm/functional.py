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
    x : torch.tensor, shape=(batch_size, 1)
    k : torch.tensor, shape=(batch_size, len(order))
    eq : torch.tensor, shape=(batch_size, len(order))
    order : int or list of int

    Returns
    -------
    u : torch.tensor, shape=(batch_size, 1)
    """
    if isinstance(order, list):
        order = torch.tensor(order)

    return torch.sum(k * (x - eq) ** order, dim=-1, keepdim=True)


def periodic(x, k, eq, order):
    """ Periodic term.

    Parameters
    ----------
    x : torch.tensor, shape=(batch_size, 1)
    k : torch.tensor, shape=(batch_size, 1)
    eq : torch.tensor, shape=(batch_size, 1)
    order : int or list of int

    Returns
    -------
    u: torch.tensor, shape=(batch_size, 1)
    """
    if isinstance(order, list):
        order = torch.tensor(order)

    return torch.sum(k * (1.0 + torch.cos(order * x - eq)), dim=-1, keepdim=True)


def lj(x, k, eq, order=torch.tensor([12, 6])):
    r""" Lennard-Jones term.

    $$
    
    E  = \epsilon  ((\sigma / r) ^ {12} - (\sigma / r) ^ 6)

    $$

    Parameters
    ----------
    x : torch.tensor, shape=(batch_size, 1)
    k : torch.tensor, shape=(batch_size, 1)
        correspond to epsilon
    eq : torch.tensor, shape=(batch_size, 1)
        correspond to sigma
    order : list of int

    Returns
    -------
    u: torch.tensor, shape=(batch_size, 1)
  
    """
    if isinstance(order, list):
        order = torch.tensor(order)

    assert order.shape[0] == 2
    assert order.dim() == 1

    return k * ((eq / x) ** order[0] - (eq / x) ** order[1])
