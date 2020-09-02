# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def periodic_torsion(x, k, 
        periodicity=list(range(1, 7)), 
        phases=[0.0 for _ in range(6)]
    ):
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
    # NOTE:
    # 0.5 because all torsions are calculated twice
    return 0.5 * esp.mm.functional.periodic(
            x=x,
            k=k,
            periodicity=periodicity,
            phases=phases,
    )
