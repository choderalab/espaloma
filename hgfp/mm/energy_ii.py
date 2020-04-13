# =============================================================================
# imports
# =============================================================================
import torch

# =============================================================================
# module functions
# =============================================================================
def bond(x, k, eq, degrees=torch.tensor([2, 3, 4])):
    # x.shape = (n_bonds, )
    # k.shape = (n_bonds, d)
    # eq.shape = (n_bonds, d)

    # (n_bonds, d)
    x_minus_eq = x - eq

    # (n_bond, )
    return torch.sum(
            torch.pow(
                x_minus_eq[:, None],
                degrees[None, :]),
            dim=1)

def angle(x, k, eq, degrees=torch.tensor([2, 3, 4])):
    # x.shape = (n_angles, )
    # k.shape = (n_angles, d)
    # eq.shape = (n_angles, d)

    # (n_angles, d)
    x_minus_eq = x - eq

    # (n_angles, )
    return torch.sum(
            torch.pow(
                x_minus_eq[:, None],
                degrees[None, :]),
            dim=1)

def torsion(x, k):
    # x.shape = (n_angles, )
    # k.shape = (n_angles, d)

    # (n_angles, d)
    x_poly = x[:, None] * torch.arange(1, k.shape[1] + 1)[None, :]

    # (n_angles, d)
    cos_x_poly = torch.cos(x_poly)

    # (n_angles, )
    return torch.sum((1 - cos_x_poly) * k, dim=1)


def lj(
        x,
        sigma_pair,
        epsilon_pair,
        switch=1.0,
        damping=0.0):

    sigma_over_r = torch.where(
        torch.gt(x, switch),
        sigma_pair / (x + damping),
        torch.zeros(x.size()))

    return epsilon_pair * (2 * torch.pow(sigma_over_r, 9) - 3 * torch.pow(sigma_over_r, 6))

def coulomb(x, q_pair):
    return torch.div(q_pair, x)

def bond_bond(x_0, x_1, k, eq_0, eq_1):
    return k * (x_0 - eq_0) * (x_1 - eq_1)

def angle_angle(x_0, x_1, k, eq_0, eq_1):
    return k * (x_0 - eq_0) * (x_1 - eq_1)

def bond_angle(x_0, x_1, k, eq_0, eq_1):
    return k * (x_0 - eq_0) * (x_1 - eq_1)

def torsion_bond(x_torsion, x_bond, k, eq):
    
    # (n_angles, )
    x_minus_eq = x_bond - eq

    # (n_angles, d)
    x_poly = x_torsion[:, None] * torch.arange(1, k.shape[1] + 1)[None, :]

    # (n_angles, d)
    cos_x_poly = torch.cos(x_poly)

    return x_minus_eq * torch.sum(k * cos_x_poly, dim=1)

def torsion_angle(x_torsion, x_angle, k, eq):
 
    # (n_angles, )
    x_minus_eq = x_angle - eq

    # (n_angles, d)
    x_poly = x_torsion[:, None] * torch.arange(1, k.shape[1] + 1)[None, :]

    # (n_angles, d)
    cos_x_poly = torch.cos(x_poly)

    return x_minus_eq * torch.sum(k * cos_x_poly, dim=1)

def torsion_angle_angle(x_torsion, x_0, x_1, k, eq_0, eq_1):
    return k * torch.cos(x_torsion) * (x_0 - eq_0) * (x_1 - eq_1)
