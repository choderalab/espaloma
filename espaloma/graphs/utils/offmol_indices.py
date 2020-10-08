import numpy as np
from openforcefield.topology import Molecule


def bond_indices(offmol: Molecule) -> np.ndarray:
    return np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])


def angle_indices(offmol: Molecule) -> np.ndarray:
    return np.array(sorted([tuple([atom.molecule_atom_index for atom in angle]) for angle in offmol.angles]))


def proper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(sorted([tuple([atom.molecule_atom_index for atom in proper]) for proper in offmol.propers]))


def improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(sorted([tuple([atom.molecule_atom_index for atom in improper]) for improper in offmol.impropers]))


def duplicate_index_ordering(indices: np.ndarray) -> np.ndarray:
    """For every (a,b,c,d) add a (d,c,b,a)

    >>> indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    >>> duplicate_index_ordering(indices)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4],
           [3, 2, 1, 0],
           [4, 3, 2, 1]])
    """
    return np.vstack([indices, indices[:, ::-1]])
