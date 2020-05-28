from typing import Iterable

import numpy as np
from openforcefield.topology import Molecule, Atom


# baseline rule-based classifiers
def get_elements(mol: Molecule, atom_indices: Iterable[Iterable[int]]) -> np.ndarray:
    """array of atomic numbers within a molecule"""
    elements = np.array([a.element.atomic_number for a in mol.atoms])
    return elements[atom_indices]


def get_element_tuples(mol: Molecule, atom_indices: Iterable[Iterable[int]]) -> list:
    """the same tuple regardless of whether atom_indices are running forward or backward"""
    return [min(tuple(t), tuple(t[::-1])) for t in get_elements(mol, atom_indices)]


is_carbon = lambda atom: atom.atomic_number == 6
is_hydrogen = lambda atom: atom.atomic_number == 1
is_oxygen = lambda atom: atom.atomic_number == 8

neighboring_carbons = lambda atom: list(filter(is_carbon, atom.bonded_atoms))
neighboring_hydrogens = lambda atom: list(filter(is_hydrogen, atom.bonded_atoms))
neighboring_oxygens = lambda atom: list(filter(is_oxygen, atom.bonded_atoms))


## atoms
def classify_atom(atom: Atom) -> int:
    if is_hydrogen(atom):
        carbon_neighborhood = neighboring_carbons(atom)
        if len(carbon_neighborhood) == 1:
            N = len(neighboring_oxygens(carbon_neighborhood[0]))
            if N >= 3:
                return 5
            elif N == 2:
                return 4
            elif N == 1:
                return 3
            else:
                return 2
        else:
            return 12
    elif is_carbon(atom):
        return 16
    elif is_oxygen(atom):
        if len(neighboring_hydrogens(atom)) == 1:
            return 19
        else:
            return 18


def classify_atoms(mol: Molecule, atom_inds: np.ndarray) -> np.ndarray:
    assert (atom_inds.shape)[1] == 1
    atoms = mol.atoms
    return np.array([classify_atom(atoms[i]) for i in atom_inds[:, 0]])


## bonds
def classify_bond(atom1: Atom, atom2: Atom) -> int:
    # both carbon
    if is_carbon(atom1) and is_carbon(atom2):
        return 1

    # one is carbon and the other oxygen, regardless of order
    elif (is_carbon(atom1) and is_oxygen(atom2)) or (is_carbon(atom2) and is_oxygen(atom1)):

        # need to catch *which* one was oxygen
        if is_oxygen(atom1):
            oxygen = atom1
        else:
            oxygen = atom2

        H = len(neighboring_hydrogens(oxygen))
        X = len(list(oxygen.bonded_atoms))

        if (X == 2) and (H == 0):
            return 15
        else:
            return 14

    # one is carbon and the other hydrogen, regardless of order
    elif (is_carbon(atom1) and is_hydrogen(atom2)) or (is_carbon(atom2) and is_hydrogen(atom1)):
        return 83

    # both oxygen
    elif is_oxygen(atom1) and is_oxygen(atom2):
        return 40

    # oxygen-hydrogen
    else:
        return 87


def classify_bonds(mol: Molecule, bond_inds: Iterable) -> np.ndarray:
    atoms = list(mol.atoms)
    return np.array([classify_bond(atoms[i], atoms[j]) for (i, j) in bond_inds])


## angles
def classify_angle(atom1: Atom, atom2: Atom, atom3: Atom) -> int:
    if is_hydrogen(atom1) and is_hydrogen(atom3):
        return 2
    elif is_oxygen(atom2):
        return 27
    else:
        return 1


def classify_angles(mol: Molecule, angle_inds: Iterable[Iterable[int]]) -> np.ndarray:
    return np.array([
        classify_angle(mol.atoms[i], mol.atoms[j], mol.atoms[k]) for (i, j, k) in angle_inds])


## torsions
# simple torsion classifier: look at element identities of atom1, atom2, atom3, and atom4
torsion_prediction_dict = {
    (6, 6, 6, 8): 1, (1, 6, 6, 6): 4, (1, 8, 6, 6): 85, (6, 6, 8, 6): 87,
    (6, 8, 6, 8): 89, (1, 6, 8, 6): 86, (8, 6, 6, 8): 5, (1, 6, 6, 8): 9,
    (1, 8, 6, 8): 84, (1, 6, 6, 1): 3, (1, 6, 8, 1): 84, (6, 6, 6, 6): 2,
    (6, 6, 8, 8): 86, (6, 8, 8, 6): 116, (1, 6, 8, 8): 86, (8, 6, 8, 8): 86,
    (6, 8, 8, 8): 116
}


def classify_torsions(mol: Molecule, torsion_inds: Iterable[Iterable[int]]):
    return np.array([torsion_prediction_dict[t] for t in get_element_tuples(mol, torsion_inds)])


if __name__ == '__main__':
    from pickle import load

    with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
        mols = load(f)

    label_dict = np.load('AlkEthOH_rings.npz')

    for type_name, classifier in [
        ('atom', classify_atoms),
        ('bond', classify_bonds),
        ('angle', classify_angles),
        ('torsion', classify_torsions)
    ]:
        n_correct, n_total = 0, 0
        for name in mols:
            mol = mols[name]
            inds, labels = label_dict[f'{name}_{type_name}_inds'], label_dict[f'{name}_{type_name}_labels']

            n_correct += sum(classifier(mols[name], inds) == labels)
            n_total += len(labels)
        print(f'{type_name}s:\n\t# correct: {n_correct}\n\t# total: {n_total}\n\taccuracy: ' + '{:.4f}%'.format(
            100 * n_correct / n_total))
