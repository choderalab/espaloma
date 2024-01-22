# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def harmonic_bond(x, k, eq):
    """Harmonic bond energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        bond length
    k : `torch.Tensor`, `shape = (batch_size, 1)`
        force constant
    eq : `torch.Tensor`, `shape = (batch_size, 1)`
        equilibrium bond length

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        energy

    """
    # NOTE:
    # the constant is included here but not in the functional forms

    # NOTE:
    # 0.25 because all bonds are calculated twice
    return 0.5 * esp.mm.functional.harmonic(x=x, k=k, eq=eq)


def harmonic_bond_mmff(x, k, eq):
    """Harmonic bond energy.

    Parameters
    ----------
    x : `torch.Tensor`, `shape = (batch_size, 1)`
        bond length
    k : `torch.Tensor`, `shape = (batch_size, 1)`
        force constant
    eq : `torch.Tensor`, `shape = (batch_size, 1)`
        equilibrium bond length

    Returns
    -------
    u : `torch.Tensor`, `shape = (batch_size, 1)`
        energy

    """
    # NOTE:
    # the constant is included here but not in the functional forms

    return (143.9325 * esp.mm.functional.quartic_expansion(x=x, k=k, eq=eq)).float()




def gaussian_bond(x, coefficients):
    """Bond energy with Gaussian basis function."""
    return esp.mm.functional.gaussian(
        x=x,
        coefficients=coefficients,
    )


def linear_mixture_bond(x, coefficients, phases):
    """Bond energy with Linear basis function.

    Parameters
    ----------
    coefficients : torch.Tensor
        Coefficients of the linear mixuture.

    phases : torch.Tensor
        Phases of the linear mixture.

    """
    return 0.5 * esp.mm.functional.linear_mixture(
        x=x, coefficients=coefficients, phases=phases
    )


def bond_high(u_bond, k3, k4):
    u_bond = u_bond - u_bond.min(dim=-1, keepdims=True)[0]
    return k3 * u_bond**1.5 + k4 * u_bond**2
