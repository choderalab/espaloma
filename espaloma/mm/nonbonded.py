# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp

# =============================================================================
# CONSTANTS
# =============================================================================
K_E = 332.0636  # kcal angstrom / (mol e ** 2)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def lj_12_6(x, k, eq):
    """ Lenard-Jones 12-6.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    k : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    eq : `torch.Tensor`,
        `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    """

    return esp.mm.functional.lj(x, k)


def columb(x, q_prod, k_e=K_E):
    """ Columb interaction without cutoff.

    Parameters
    ----------
    x : `torch.Tensor`, shape=`(batch_size, 1)` or `(batch_size, batch_size, 1)`
    q_prod : `torch.Tensor`,
        `shape=(batch_size, 1) or `(batch_size, batch_size, 1)`

    Returns
    -------
    u : `torch.Tensor`,
        `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`


    """
    return k_e * x / q_prod
