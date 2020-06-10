# separate parameters for every atom, bond, angle, torsion, up to symmetry

import numpy as np
from espaloma.data.alkethoh.data import offmols
from openeye import oechem
from openforcefield.topology import Molecule
from tqdm import tqdm


def atom_symmetry_classes(offmol: Molecule):
    oemol = offmol.to_openeye()
    oechem.OEPerceiveSymmetry(oemol)
    symmetry_classes = np.array([atom.GetSymmetryClass() for atom in oemol.GetAtoms()])
    return symmetry_classes


n_unique = 0
n_total = 0

symmetry_classes = {}

# Atom types
for name in tqdm(offmols):
    offmol = offmols[name]
    symmetry_classes[name] = atom_symmetry_classes(offmol)
    if offmol.n_atoms != len(symmetry_classes[name]):
        print(f'{offmol.n_atoms} != {len(symmetry_classes[name])}')

    n_unique += len(set(symmetry_classes[name]))
    n_total += offmol.n_atoms

print(f'atoms: {n_unique} / {n_total} = {n_unique / n_total:.3f}')


def canonicalize_order(tup):
    return min(tup, tup[::-1])


def get_unique_bonds(offmol):
    """
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
    """
    triple_inds:
        array of shape (n_angles, 3)
    angle_inds:
        array of shape (n_angles,)
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
    offmol = offmols[name]
    torsions = offmol.propers
    sym = symmetry_classes[name]

    torsion_tups = []

    for torsion in torsions:
        # is this off by one?
        tup = tuple(sym[atom.molecule_atom_index] for atom in torsion)
        torsion_tups.append(canonicalize_order(tup))
    n_unique += len(set(torsion_tups))
    n_total += len(torsion_tups)
print(f'torsions: {n_unique} / {n_total} = {n_unique / n_total:.3f}')


# TODO: make the import structure clearer
from espaloma.data.alkethoh.mm_utils import harmonic_bond_potential, harmonic_angle_potential, periodic_torsion_potential
from espaloma.data.alkethoh.neural_baseline import extract_bond_term_inputs, compute_distances, compute_angles, compute_torsions
from espaloma.data.alkethoh.neural_baseline import get_snapshots_and_energies

def compute_harmonic_bond_potential(offmol, xyz, params, bond_inds):
    """

    :param offmol:
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param bond_inds:
        numpy array of length offmol.n_atoms,
        taking integer values in range 1 - n_unique
    :return:
    """


    n_unique = int(len(params)/2)
    ks, r0s = params[:n_unique], params[n_unique:]
    k, r0 = ks[bond_inds], r0s[bond_inds]

    x, inds = extract_bond_term_inputs(offmol)
    r = compute_distances(xyz, inds)
    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)





if __name__ == '__main__':
    xyz, _ = get_snapshots_and_energies()



    compute_harmonic_bond_potential(offmol, )
