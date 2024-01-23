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
    """Periodic torsion potential

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
        x=x,
        k=k,
        periodicity=periodicity,
        phases=phases,
    )
    # assert(out.shape == (len(x), 1))
    return out

def periodic_torsion_mmff(
    x, k, periodicity=list(range(1, 4))
):
    """Periodic torsion potential

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
    out = 0.5 * esp.mm.functional.periodic_mmff(
        x=x,
        k=k,
        periodicity=periodicity,
    )
    # assert(out.shape == (len(x), 1))
    return out



def angle_angle(
    u_angle_left,
    u_angle_right,
    k_angle_angle,
):

    u_angle_left = u_angle_left - u_angle_left.min(dim=-1, keepdims=True)[0]
    u_angle_right = (
        u_angle_right - u_angle_right.min(dim=-1, keepdims=True)[0]
    )
    return k_angle_angle * (u_angle_left**0.5) * (u_angle_right**0.5)


def angle_torsion(
    u_angle_left,
    u_angle_right,
    u_torsion,
    k_angle_torsion,
):
    u_angle_left = u_angle_left - u_angle_left.min(dim=-1, keepdims=True)[0]
    u_angle_right = (
        u_angle_right - u_angle_right.min(dim=-1, keepdims=True)[0]
    )
    return (
        k_angle_torsion * (u_angle_left**0.5) * u_torsion
        + k_angle_torsion * (u_angle_right**0.5) * u_torsion
    )


def angle_angle_torsion(
    u_angle_left,
    u_angle_right,
    u_torsion,
    k_angle_angle_torsion,
):
    u_angle_left = u_angle_left - u_angle_left.min(dim=-1, keepdims=True)[0]
    u_angle_right = (
        u_angle_right - u_angle_right.min(dim=-1, keepdims=True)[0]
    )
    return (
        k_angle_angle_torsion
        * (u_angle_left**0.5)
        * (u_angle_right**0.5)
        * u_torsion
    )


def bond_torsion(
    u_bond_left,
    u_bond_right,
    u_bond_center,
    u_torsion,
    k_side_torsion,
    k_center_torsion,
):

    u_bond_left = u_bond_left - u_bond_left.min(dim=-1, keepdims=True)[0]
    u_bond_right = u_bond_right - u_bond_right.min(dim=-1, keepdims=True)[0]
    u_bond_center = (
        u_bond_center - u_bond_center.min(dim=-1, keepdims=True)[0]
    )
    return (
        k_side_torsion * u_torsion * (u_bond_left**0.5)
        + k_side_torsion * u_torsion * (u_bond_right**0.5)
        + k_center_torsion * u_torsion * (u_bond_center**0.5)
    )
