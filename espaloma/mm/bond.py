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
        eqilibrium value

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
#
# def harmonic_bond_re(x, k, eq):
#     return 0.25 * esp.mm.functional.harmonic_re(x=x, k=k, eq=eq)
