# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic_bond(x, k, eq):
    """ Harmonic bond energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        bond length
    k : `torch.Tensor`, `shape = (batch_size, 1)`
        force constant
    eq : `torch.Tensor`, `shape = (batch_size, 1)`
        equilibrium bond length

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        energy

    """
    # NOTE:
    # the constant is included here but not in the functional forms

    # NOTE:
    # 0.25 because all bonds are calculated twice
    return 0.25 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)


def gaussian_bond(x, coefficients):
    """ Bond energy with Gaussian basis function.

    """
    return esp.mm.functional.gaussian(x=x, coefficients=coefficients,)


def linear_mixture_bond(x, coefficients):
    """ Bond energy with Linear basis function.

    """
    return esp.mm.functional.linear_mixture(x=x, coefficients=coefficients,)
