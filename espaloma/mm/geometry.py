# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def distance(x0, x1):
    """ Distance. """
    return torch.norm(x0 - x1, p=2, dim=-1)


def _angle(r0, r1):
    """ Angle between vectors. """

    return torch.atan2(
        torch.norm(torch.cross(r0, r1), p=2, dim=-1),
        torch.sum(torch.mul(r0, r1), dim=-1),
    )


def angle(x0, x1, x2):
    """ Angle between three points. """
    left = x1 - x0
    right = x1 - x2
    return _angle(left, right)


def _dihedral(r0, r1):
    """ Dihedral between normal vectors. """
    return _angle(r0, r1)


def dihedral(x0, x1, x2, x3):
    """ Dihedral between four points. """
    left = torch.cross(x1 - x0, x1 - x2)
    right = torch.cross(x2 - x1, x2 - x3)
    return _dihedral(left, right)
