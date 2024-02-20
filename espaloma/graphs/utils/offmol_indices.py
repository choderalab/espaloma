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
    """ "[*:1]~[*:2](~[*:3])~[*:4]" matches"""

    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in improper])
                for improper in offmol.impropers
            ]
        )
    )


def improper_torsion_indices(
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

    ## Find all atoms bound to exactly 3 other atoms
    if improper_def == "espaloma":
        ## This finds all orderings, which is what we want for the espaloma case
        ##  but not for smirnoff
        improper_smarts = "[*:1]~[X3:2](~[*:3])~[*:4]"
        mol_idxs = offmol.chemical_environment_matches(improper_smarts)
        return np.array(mol_idxs)
    elif improper_def == "smirnoff":
        improper_smarts = "[*:2]~[X3:1](~[*:3])~[*:4]"
        ## For smirnoff ordering, we only want to find the unique combinations
        ##  of atoms forming impropers so we can permute them the way we want
        mol_idxs = offmol.chemical_environment_matches(
            improper_smarts, unique=True
        )

        ## Get all ccw orderings
        # feels like there should be some good way to do this with itertools...
        idx_permuts = []
        for c, *other_atoms in mol_idxs:
            for i in range(3):
                idx = [c]
                for j in range(3):
                    idx.append(other_atoms[(i + j) % 3])
                idx_permuts.append(tuple(idx))

        return np.array(idx_permuts)
    else:
        raise ValueError(f"Unknown value for improper_def: {improper_def}")


def out_of_plane_indices(
    offmol: Molecule
) -> np.ndarray:
    """ "[*:1]~[X3:2](~[*:3])~[*:4]" matches (_all_improper_torsion_indices returns "[*:1]~[*:2](~[*:3])~[*:4]" matches)

    improper_def allows for choosing which atom will be the central atom in the
    permutations:
    smirnoff: central atom is listed first
    

    Addtionally, for smirnoff, only take the subset of atoms that corresponds
    to the ccw traversal of connected atoms.

    Notes
    -----
    Motivation: offmol.impropers returns a large number of impropers, and we may wish to restrict this number.
    May update this filter definition based on discussion in https://github.com/openff.toolkit/openff.toolkit/issues/746
    """
    improper_smarts = "[*:1]~[X3:2](~[*:3])~[*:4]"
    mol_idxs = offmol.chemical_environment_matches(improper_smarts)
    return np.array(mol_idxs)
