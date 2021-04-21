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


def angle_angle(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        k_angle_angle,
    ):
    return esp.mm.functional.harmonic_harmonic_coupled(
        x0=x_angle_left,
        eq0=eq_angle_left,
        x1=x_angle_right,
        eq1=eq_angle_right,
        k=k_angle_angle,
    )

def angle_torsion(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        x,
        k_angle_torsion_left,
        k_angle_torsion_right,
        periodicity=list(range(1,3)),
    ):
    return esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_angle_left,
        x_periodic=x,
        k=k_angle_torsion_left,
        eq=eq_angle_left,
        periodicity=periodicity,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_angle_right,
        x_periodic=x,
        k=k_angle_torsion_right,
        eq=eq_angle_right,
        periodicity=periodicity,
    )

def angle_angle_torsion(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        x,
        k_angle_angle_torsion,
    ):
    return esp.mm.functional.harmonic_harmonic_periodic_coupled(
        theta0=x_angle_left,
        theta1=x_angle_right,
        eq0=eq_angle_left,
        eq1=eq_angle_right,
        phi=x,
        k=k_angle_angle_torsion,
    )

def bond_torsion(
        x_bond_left,
        x_bond_center,
        x_bond_right,
        x,
        k_left_torsion,
        k_center_torsion,
        k_right_torsion,
        eq_left_torsion,
        eq_center_torsion,
        eq_right_torsion,
    ):
    return esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_left,
        x_periodic=x,
        k=k_left_torsion,
        eq=eq_left_torsion,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_center,
        x_periodic=x,
        k=k_center_torsion,
        eq=eq_center_torsion,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_right,
        x_periodic=x,
        k=k_right_torsion,
        eq=eq_right_torsion,
    )

def angle_angle(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        k_angle_angle,
    ):
    return esp.mm.functional.harmonic_harmonic_coupled(
        x0=x_angle_left,
        eq0=eq_angle_left,
        x1=x_angle_right,
        eq1=eq_angle_right,
        k=k_angle_angle,
    )

def angle_torsion(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        x,
        k_angle_torsion_left,
        k_angle_torsion_right,
        periodicity=list(range(1,3)),
    ):
    return esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_angle_left,
        x_periodic=x,
        k=k_angle_torsion_left,
        eq=eq_angle_left,
        periodicity=periodicity,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_angle_right,
        x_periodic=x,
        k=k_angle_torsion_right,
        eq=eq_angle_right,
        periodicity=periodicity,
    )

def angle_angle_torsion(
        x_angle_left,
        x_angle_right,
        eq_angle_left,
        eq_angle_right,
        x,
        k_angle_angle_torsion,
    ):
    return esp.mm.functional.harmonic_harmonic_periodic_coupled(
        theta0=x_angle_left,
        theta1=x_angle_right,
        eq0=eq_angle_left,
        eq1=eq_angle_right,
        phi=x,
        k=k_angle_angle_torsion,
    )

def bond_torsion(
        x_bond_left,
        x_bond_center,
        x_bond_right,
        x,
        k_left_torsion,
        k_center_torsion,
        k_right_torsion,
        eq_left_torsion,
        eq_center_torsion,
        eq_right_torsion,
    ):
    return esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_left,
        x_periodic=x,
        k=k_left_torsion,
        eq=eq_left_torsion,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_center,
        x_periodic=x,
        k=k_center_torsion,
        eq=eq_center_torsion,
    ) + esp.mm.functional.harmonic_periodic_coupled(
        x_harmonic=x_bond_right,
        x_periodic=x,
        k=k_right_torsion,
        eq=eq_right_torsion,
    )
