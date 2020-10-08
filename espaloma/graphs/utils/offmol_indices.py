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
