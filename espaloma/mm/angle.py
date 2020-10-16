# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic_angle(x, k, eq):
    """ Harmonic angle energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        angle value
    k : `torch.Tensor`, `shape = (batch_size, 1)`
        force constant
    eq : `torch.Tensor`, `shape = (batch_size, 1)`
        equilibrium angle

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        energy

    """
    # NOTE:
    # the constant 0.5 is included here but not in the functional forms

    # NOTE:
    # 0.25 because all angles are calculated twice
    return 0.25 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)
