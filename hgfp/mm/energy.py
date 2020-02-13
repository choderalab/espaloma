# =============================================================================
# imports
# =============================================================================
import torch

# =============================================================================
# module functions
# =============================================================================
def bond(x, k, l):

    return 0.5 * k * (x - l) ** 2

def angle(x, k, l):
    return 0.5 * k * (x - l) ** 2
    return torch.mul(
        torch.mul(
            0.5,
            k),
        tf.math.square(
            tf.math.subtract(
                x,
                l)))

def torsion(x, k, l):
    return k * (1 + torch.cos(x - l))

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
