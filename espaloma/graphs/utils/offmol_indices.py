import numpy as np
from openforcefield.topology import Molecule


def atom_indices(offmol: Molecule) -> np.ndarray:
    return np.array([a.molecule_atom_index for a in offmol.atoms])


def bond_indices(offmol: Molecule) -> np.ndarray:
    return np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])


def angle_indices(offmol: Molecule) -> np.ndarray:
    return np.array(sorted([tuple([atom.molecule_atom_index for atom in angle]) for angle in offmol.angles]))


def proper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(sorted([tuple([atom.molecule_atom_index for atom in proper]) for proper in offmol.propers]))


def _all_improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    """"[*:1]~[*:2](~[*:3])~[*:4]" matches"""

    return np.array(sorted([tuple([atom.molecule_atom_index for atom in improper]) for improper in offmol.impropers]))


def improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    """"[*:1]~[X3:2](~[*:3])~[*:4]" matches (_all_improper_torsion_indices returns "[*:1]~[*:2](~[*:3])~[*:4]" matches)

    Notes
    -----
    Motivation: offmol.impropers returns a large number of impropers, and we may wish to restrict this number.
    May update this filter definition based on discussion in https://github.com/openforcefield/openforcefield/issues/746
    """
    improper_smarts = "[*:1]~[X3:2](~[*:3])~[*:4]"
    return np.array(offmol.chemical_environment_matches(improper_smarts))
