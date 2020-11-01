# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def periodic_torsion(
    x, k, periodicity=list(range(1, 7)), phases=[0.0 for _ in range(6)]
):
    """ Periodic torsion potential

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        Dihedral value.
    k : `torch.Tensor`, `shape = (batch_size, n_phases)`
        Force constants.
    periodicity : `torch.Tensor`, `shape = (batch_size, n_phases)`
        Periodicities
    phases : `torch.Tensor`, `shape = (batch_size, n_phases)`
        Phase offsets

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        Energy.

    """

    # NOTE:
    # 0.5 because all torsions are calculated twice
    out = 0.5 * esp.mm.functional.periodic(
        x=x, k=k, periodicity=periodicity, phases=phases,
    )
    # assert(out.shape == (len(x), 1))
    return out
