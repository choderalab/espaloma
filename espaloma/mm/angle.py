# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic_angle(x, k, eq):
    """ Harmonic bond energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        angle value
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
    # the constant 0.5 is included here but not in the functional forms

    # NOTE:
    # 0.25 because all angles are calculated twice
    return 0.25 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)

def urey_bradley(x_between, k_urey_bradley, eq_urey_bradley):
    return esp.mm.function.harmonic(
        x=x_between,
        k=k_urey_bradley,
        eq=eq_urey_bradley,
    )

def bond_bond(x_left, x_right, eq_left, eq_right, k_bond_bond):
    return esp.mm.function.harmonic_harmonic_coupled(
        x0=x_left,
        x1=x_right,
        eq0=eq_left,
        eq1=eq_right,
        k=k_bond_bond,
    )

def bond_angle(
        x_left, x_right, x_angle, eq_left, eq_right, eq_angle,
        k_bond_angle_left, k_bond_angle_right,
    ):
    return esp.mm.function.harmonic_harmonic_coupled(
        x0=x_left,
        x1=x_angle,
        eq0=eq_left,
        eq1=eq_angle,
        k=k_bond_angle_left,
    ) + esp.mm.function.harmonic_harmonic_coupled(
        x0=x_right,
        x1=x_angle,
        eq0=eq_right,
        eq1=eq_angle,
        k=k_bond_angle_right,
    )
