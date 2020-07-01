# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def periodic_torsion(x, k, eq):
    """ Harmonic bond energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        Dihedral value.
    k : `torch.Tensor`, `shape = (batch_size, periodicity)`
        Force constants.
    eq : `torch.Tensor`, `shape = (batch_size, periodicity)`
        Phase offset.

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        Energy.

    """
    return esp.mm.functional.periodic(x, k, eq)
