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
    x : torch.Tensor, shape = (batch_size, 1)
        bond length
    k : torch.Tensor, shape = (batch_size, 1)
        force constant
    eq : torch.Tensor, shape = (batch_size, 1)
        eqilibrium value

    Returns
    -------
    u : torch.tensor, shape = (batch_size, 1)
        energy

    """
    # NOTE:
    # the constant 0.5 is included here but not in the functional forms
    return 0.5 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)
