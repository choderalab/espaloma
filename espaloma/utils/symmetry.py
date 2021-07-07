import jax.numpy as np
import numpy as onp
from openeye import oechem
from openforcefield.topology import Molecule


def atom_symmetry_classes(offmol: Molecule):
    """return integer array of length offmol.n_atoms, labeling symmetry class of each atom"""
    oemol = offmol.to_openeye()
    oechem.OEPerceiveSymmetry(oemol)
    symmetry_classes = onp.array([atom.GetSymmetryClass() for atom in oemol.GetAtoms()])
    return symmetry_classes


def canonicalize_order(tup):
    """want to treat (a,b,c,d) same as (d,c,b,a), so return min((a,b,c,d), (d,c,b,a))"""
    return min(tup, tup[::-1])


# TODO: def get_unique_atoms(offmol)?

def get_unique_bonds(offmol: Molecule):
    """based on symmetry classes of atoms, identify symmetry classes of bonds

    returns
    pair_inds:
        array of shape (n_bonds, 2)
    bond_inds:
        array of shape (n_bonds,)
    """

    sym = atom_symmetry_classes(offmol)

    pair_inds = []
    bond_tups = []

    for bond in offmol.bonds:
        pair_inds.append((bond.atom1_index, bond.atom2_index))
        tup = (sym[bond.atom1_index], sym[bond.atom2_index])
        bond_tups.append(canonicalize_order(tup))

    pair_inds = np.array(pair_inds)

    bond_set = set(bond_tups)
    bond_ind_map = dict(zip(bond_set, range(len(bond_set))))
    bond_inds = np.array([bond_ind_map[tup] for tup in bond_tups])

    return pair_inds, bond_inds


def get_unique_angles(offmol):
    """based on symmetry classes of atoms, identify symmetry classes of angles

    returns
    triple_inds:
        array of shape (n_angles, 3)
    angle_inds:
        array of shape (n_angles,)

    TODO: refactor to avoid code-duplication between bond, angle, tuple
    """

    sym = atom_symmetry_classes(offmol)

    triple_inds = []
    angle_tups = []

    for angle in offmol.angles:
        triple_inds.append(tuple((atom.molecule_atom_index for atom in angle)))
        tup = tuple(sym[atom.molecule_atom_index] for atom in angle)
        angle_tups.append(canonicalize_order(tup))

    triple_inds = np.array(triple_inds)

    angle_set = set(angle_tups)
    angle_ind_map = dict(zip(angle_set, range(len(angle_set))))
    angle_inds = np.array([angle_ind_map[tup] for tup in angle_tups])

    return triple_inds, angle_inds


def get_unique_torsions(offmol):
    """based on symmetry classes of atoms, identify symmetry classes of torsions

    returns
    quad_inds:
        array of shape (n_angles, 4)
    torsion_inds:
        array of shape (n_angles,)

    TODO: refactor to avoid code-duplication between bond, angle, tuple
    """

    sym = atom_symmetry_classes(offmol)

    quad_inds = []
    torsion_tups = []

    for torsion in offmol.propers:
        quad_inds.append(tuple((atom.molecule_atom_index for atom in torsion)))
        tup = tuple(sym[atom.molecule_atom_index] for atom in torsion)
        torsion_tups.append(canonicalize_order(tup))

    quad_inds = np.array(quad_inds)

    torsion_set = set(torsion_tups)
    torsion_ind_map = dict(zip(torsion_set, range(len(torsion_set))))
    torsion_inds = np.array([torsion_ind_map[tup] for tup in torsion_tups])

    return quad_inds, torsion_inds


if __name__ == '__main__':
    from tqdm import tqdm
    from espaloma.data.alkethoh.data import offmols

    # Atom types
    n_unique = 0
    n_total = 0

    symmetry_classes = {}
    for name in tqdm(offmols):
        offmol = offmols[name]
        symmetry_classes[name] = atom_symmetry_classes(offmol)
        if offmol.n_atoms != len(symmetry_classes[name]):
            print(f'{offmol.n_atoms} != {len(symmetry_classes[name])}')

        n_unique += len(set(symmetry_classes[name]))
        n_total += offmol.n_atoms

    print(f'atoms: {n_unique} / {n_total} = {n_unique / n_total:.3f}')

    # Bond types
    n_unique = 0
    n_total = 0
    for name in tqdm(offmols):
        pair_inds, bond_inds = get_unique_bonds(offmols[name])

        n_unique += len(set(bond_inds))
        n_total += len(bond_inds)
    print(f'bonds: {n_unique} / {n_total} = {n_unique / n_total:.3f}')

    # Angle types
    n_unique = 0
    n_total = 0
    for name in tqdm(offmols):
        triple_inds, angle_inds = get_unique_angles(offmols[name])

        n_unique += len(set(angle_inds))
        n_total += len(angle_inds)
    print(f'angles: {n_unique} / {n_total} = {n_unique / n_total:.3f}')

    # Torsion types
    n_unique = 0
    n_total = 0
    for name in tqdm(offmols):
        quad_inds, torsion_inds = get_unique_torsions(offmols[name])

        n_unique += len(set(torsion_inds))
        n_total += len(torsion_inds)
    print(f'torsions: {n_unique} / {n_total} = {n_unique / n_total:.3f}')
