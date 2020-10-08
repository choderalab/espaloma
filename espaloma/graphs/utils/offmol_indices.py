import numpy as np
from openforcefield.topology import Molecule


def bond_indices(offmol: Molecule) -> np.ndarray:
    return np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])

# TODO: angle indices

# TODO: proper torsion indices

# TODO: improper torsion indices
