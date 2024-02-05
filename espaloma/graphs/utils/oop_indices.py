import numpy as np
from openff.toolkit.topology import Molecule


def atom_indices(offmol: Molecule) -> np.ndarray:
    return np.array([a.molecule_atom_index for a in offmol.atoms])


def bond_indices(offmol: Molecule) -> np.ndarray:
    return np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])


def angle_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in angle])
                for angle in offmol.angles
            ]
        )
    )



def out_of_plane_indices(
    offmol: Molecule, improper_def="espaloma"
) -> np.ndarray:
    """ "[*:1]~[X3:2](~[*:3])~[*:4]" matches (_all_improper_torsion_indices returns "[*:1]~[*:2](~[*:3])~[*:4]" matches)

    improper_def allows for choosing which atom will be the central atom in the
    permutations:
    smirnoff: central atom is listed first
    espaloma: central atom is listed second

    Addtionally, for smirnoff, only take the subset of atoms that corresponds
    to the ccw traversal of connected atoms.

    Notes
    -----
    Motivation: offmol.impropers returns a large number of impropers, and we may wish to restrict this number.
    May update this filter definition based on discussion in https://github.com/openff.toolkit/openff.toolkit/issues/746
    """
    breakpoint()
    pass
