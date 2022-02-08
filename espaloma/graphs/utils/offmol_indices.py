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


def proper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in proper])
                for proper in offmol.propers
            ]
        )
    )


def _all_improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    """"[*:1]~[*:2](~[*:3])~[*:4]" matches"""

    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in improper])
                for improper in offmol.impropers
            ]
        )
    )


def improper_torsion_indices(offmol: Molecule, improper_def='espaloma') -> np.ndarray:
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
    ## Check options
    if (improper_def != 'espaloma') and (improper_def != 'smirnoff'):
        raise ValueError(f'Unknown value for improper_def: {improper_def}')

    ## Find all atoms bound to exactly 3 other atoms
    ## This finds all orderings, which is what we want for the espaloma case
    ##  but not for smirnoff
    improper_smarts = "[*:1]~[X3:2](~[*:3])~[*:4]"
    mol_idxs = offmol.chemical_environment_matches(improper_smarts)

    if improper_def == 'espaloma':
        return np.array(mol_idxs)

    ## Get all unique improper centers, and the list of atoms that they
    ##  bind to
    imp_centers = {}
    for idxs in mol_idxs:
        center_atom = idxs[1]
        if center_atom in imp_centers:
            continue
        other_atoms = tuple(sorted([idxs[0]]+list(idxs[2:])))
        imp_centers[center_atom] = other_atoms

    ## Get all ccw orderings
    # feels like there should be some good way to do this with itertools...
    mol_idxs = []
    for c, other_atoms in imp_centers.items():
        for i in range(3):
            idx = [c]
            for j in range(3):
                idx.append(other_atoms[(i+j)%3])
            mol_idxs.append(tuple(idx))

    return np.array(mol_idxs)