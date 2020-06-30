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
    x : torch.Tensor, shape = (batch_size, 1)
        dihedral value
    k : torch.Tensor, shape = (batch_size, periodicity)
        force constants
    eq : torch.Tensor, shape = (batch_size, periodicity)
        phase offset

    Returns
    -------
    u : torch.tensor, shape = (batch_size, 1)
        energy

    """
    return esp.mm.functional.periodic(x, k, eq)
