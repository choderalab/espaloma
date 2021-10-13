# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic_angle(x, k, eq):
    """Harmonic angle energy.

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
    return 0.5 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)


def linear_mixture_angle(x, coefficients, phases):
    """Angle energy with Linear basis function."""

    return 0.5 * esp.mm.functional.linear_mixture(
        x=x, coefficients=coefficients, phases=phases
    )


def urey_bradley(x_between, coefficients, phases):
    return esp.mm.functional.linear_mixture(
        x=x_between,
        coefficients=coefficients,
        phases=phases,
    )


def bond_bond(u_left, u_right, k_bond_bond):
    u_left = u_left - u_left.min(dim=-1, keepdims=True)[0]
    u_right = u_right - u_right.min(dim=-1, keepdims=True)[0]
    return k_bond_bond * (u_left ** 0.5) * (u_right ** 0.5)


def bond_angle(
    u_left,
    u_right,
    u_angle,
    k_bond_angle,
):

    u_left = u_left - u_left.min(dim=-1, keepdims=True)[0]
    u_right = u_right - u_right.min(dim=-1, keepdims=True)[0]
    u_angle = u_angle - u_angle.min(dim=-1, keepdims=True)[0]
    return k_bond_angle * (u_left ** 0.5) * (
        u_angle ** 0.5
    ) + k_bond_angle * (u_right ** 0.5) * (u_angle ** 0.5)


def angle_high(
    u_angle,
    k3,
    k4,
):
    u_angle = u_angle - u_angle.min(dim=-1, keepdims=True)[0]
    return k3 * u_angle ** 1.5 + k4 * u_angle ** 2
