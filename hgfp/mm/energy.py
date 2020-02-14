# =============================================================================
# imports
# =============================================================================
import torch

# =============================================================================
# module functions
# =============================================================================
def bond(x, k, eq):
    return 0.5 * k * (x - eq) ** 2

def angle(x, k, eq):
    return 0.5 * k * (x - eq) ** 2

def torsion(x, k, eq):
    return k * (1 + torch.cos(x - eq))

def lj(
        x,
        sigma_pair,
        epsilon_pair,
        switch=0.0,
        damping=0.0):
    """ Calculate the 12-6 Lenard Jones energy.

    """


    sigma_over_r = torch.where(
        torch.gt(x, switch),
        sigma_pair / (x + damping),
        torch.zeros(x.size()))


    return 4 * epsilon_pair * (torch.pow(sigma_over_r, 12) - torch.pow(sigma_over_r, 6))
