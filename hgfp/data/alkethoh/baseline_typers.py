import numpy as np
from hgfp.data.alkethoh.label_molecules import get_labeled_atoms

# baseline rule-based classifiers

def get_elements(mol, atom_indices):
    elements = np.array([a.element.atomic_number for a in mol.atoms])
    return elements[atom_indices]


def get_element_tuples(mol, atom_indices):
    """return the same tuple regardless of whether atom_indices are running forward or backward

    for torsions, get_element_tuples(i,j,k,l)

    """
    return [min(tuple(t), tuple(t[::-1])) for t in get_elements(mol, atom_indices)]

def is_carbon(atom):
    return atom.atomic_number == 6


def is_hydrogen(atom):
    return atom.atomic_number == 1


def is_oxygen(atom):
    return atom.atomic_number == 8


def neighboring_carbons(atom):
    return list(filter(is_carbon, atom.bonded_atoms))


def neighboring_hydrogens(atom):
    return list(filter(is_hydrogen, atom.bonded_atoms))


def neighboring_oxygens(atom):
    return list(filter(is_oxygen, atom.bonded_atoms))


def classify_atom(atom):
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


def classify_atoms(mol):
    return np.array([classify_atom(a) for a in mol.atoms])


def classify_bond(atom1, atom2):
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

def classify_bonds(mol, bond_inds):
    atoms = list(mol.atoms)
    return np.array([classify_bond(atoms[i], atoms[j]) for (i,j) in bond_inds])


def classify_angle(atom1, atom2, atom3):
    if is_hydrogen(atom1) and is_hydrogen(atom3):
        return 2
    elif is_oxygen(atom2):
        return 27
    else:
        return 1


def classify_angles(mol, angle_inds):
    return np.array([
        classify_angle(mol.atoms[i], mol.atoms[j], mol.atoms[k]) for (i, j, k) in angle_inds])


# simple torsion classifier: look at element identities of atom1, atom2, atom3, and atom4
from collections import defaultdict

def learn_torsion_lookup():
    # TODO: collect rest of missing logic...
    torsion_tuple_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(mols)):
        torsion_inds, torsion_labels = get_labeled_torsions(labeled_mols[i])
        element_tuples = get_element_tuples(mols[i], torsion_inds)
        for j in range(len(torsion_labels)):
            torsion_tuple_counts[element_tuples[j]][torsion_labels[j]] += 1



# the learned classifier
torsion_prediction_dict = {
    (6, 6, 6, 8): 1, (1, 6, 6, 6): 4, (1, 8, 6, 6): 85, (6, 6, 8, 6): 87,
    (6, 8, 6, 8): 89, (1, 6, 8, 6): 86, (8, 6, 6, 8): 5, (1, 6, 6, 8): 9,
    (1, 8, 6, 8): 84, (1, 6, 6, 1): 3, (1, 6, 8, 1): 84, (6, 6, 6, 6): 2,
    (6, 6, 8, 8): 86, (6, 8, 8, 6): 116, (1, 6, 8, 8): 86, (8, 6, 8, 8): 86,
    (6, 8, 8, 8): 116
}


def classify_torsions(mol, torsion_inds):
    return np.array([torsion_prediction_dict[t] for t in get_element_tuples(mol, torsion_inds)])


if __name__ == '__main__':
    from pickle import load
    with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
        mols = load(f)

    label_dict = np.load('AlkEthOH_rings.npz')

    # atoms
    n_correct, n_total = 0, 0
    for name in mols:
        mol = mols[name]
        inds, labels = label_dict[f'{name}_atom_inds'], label_dict[f'{name}_atom_labels']

        ## TODO: track this down
        #n_correct += sum(classify_atoms(mols[name])[inds] == labels) # problem: atom_inds and atom_labels aren't the same shape!

        # TODO: update classify_atoms signature to accept indices, just like bonds/angles/torsions
        n_correct += sum(classify_atoms(mols[name]) == labels)
        n_total += mol.n_atoms
    print('atoms: ', n_correct, n_total)

    # bonds
    n_correct, n_total = 0, 0
    for name in mols:
        mol = mols[name]
        inds, labels = label_dict[f'{name}_bond_inds'], label_dict[f'{name}_bond_labels']

        n_correct += sum(classify_bonds(mols[name], inds) == labels)
        n_total += len(labels)
    print('bonds: ', n_correct, n_total)

    # angles
    n_correct, n_total = 0, 0
    for name in mols:
        mol = mols[name]
        inds, labels = label_dict[f'{name}_angle_inds'], label_dict[f'{name}_angle_labels']

        n_correct += sum(classify_angles(mols[name], inds) == labels)
        n_total += len(labels)
    print('angles: ', n_correct, n_total)

    # torsions
    n_correct, n_total = 0, 0
    for name in mols:
        mol = mols[name]
        inds, labels = label_dict[f'{name}_torsion_inds'], label_dict[f'{name}_torsion_labels']

        n_correct += sum(classify_torsions(mols[name], inds) == labels)
        n_total += len(labels)
    print('torsions: ', n_correct, n_total)
