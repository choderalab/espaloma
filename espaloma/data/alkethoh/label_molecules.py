"""label every molecule in AlkEthOH rings set"""

from pickle import dump

import numpy as np
from espaloma.data.alkethoh.data import alkethoh_url, path_to_smiles, path_to_offmols, path_to_npz, download_alkethoh
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from tqdm import tqdm

# Load the OpenFF "Parsley" force field. -- pick unconstrained so that Hbond stretches are sampled...
forcefield = ForceField('openff_unconstrained-1.0.0.offxml')


## loading molecules
# TODO: replace mol_from_smiles with something that reads the mol2 files directly...
def mol_from_smiles(smiles):
    mol = Molecule.from_smiles(smiles, hydrogens_are_explicit=False, allow_undefined_stereo=True)
    return mol


## labeling molecules
def label_mol(mol):
    return forcefield.label_molecules(mol.to_topology())[0]


def get_inds_and_labels(labeled_mol, type_name='vdW'):
    terms = labeled_mol[type_name]
    inds = np.array(list(terms.keys()))
    labels = np.array([int(term.id[1:]) for term in terms.values()])

    assert (len(inds) == len(labels))

    return inds, labels


from functools import partial

get_labeled_atoms = partial(get_inds_and_labels, type_name='vdW')
get_labeled_bonds = partial(get_inds_and_labels, type_name='Bonds')
get_labeled_angles = partial(get_inds_and_labels, type_name='Angles')
get_labeled_torsions = partial(get_inds_and_labels, type_name='ProperTorsions')

if __name__ == '__main__':
    # download, if it isn't already present
    download_alkethoh()

    # load molecules
    with open(path_to_smiles, 'r') as f:
        smiles = [l.split()[0] for l in f.readlines()]
    with open(path_to_smiles, 'r') as f:
        names = [l.split()[1] for l in f.readlines()]

    mols = dict()
    for i in range(len(names)):
        mols[names[i]] = Molecule.from_smiles(smiles[i], allow_undefined_stereo=True)

    with open(path_to_offmols, 'wb') as f:
        dump(mols, f)

    # Label molecules using forcefield
    # Takes about ~200ms per molecule -- can do ~1000 molecules in ~5-6 minutes, sequentially
    labeled_mols = dict()
    for name in tqdm(names):
        labeled_mols[name] = label_mol(mols[name])

    label_dict = dict()
    n_atoms, n_bonds, n_angles, n_torsions = 0, 0, 0, 0

    for name in names:
        labeled_mol = labeled_mols[name]
        label_dict[f'{name}_atom_inds'], label_dict[f'{name}_atom_labels'] = get_labeled_atoms(labeled_mol)
        n_atoms += len(label_dict[f'{name}_atom_inds'])
        label_dict[f'{name}_bond_inds'], label_dict[f'{name}_bond_labels'] = get_labeled_bonds(labeled_mol)
        n_bonds += len(label_dict[f'{name}_bond_inds'])
        label_dict[f'{name}_angle_inds'], label_dict[f'{name}_angle_labels'] = get_labeled_angles(labeled_mol)
        n_angles += len(label_dict[f'{name}_angle_inds'])
        label_dict[f'{name}_torsion_inds'], label_dict[f'{name}_torsion_labels'] = get_labeled_torsions(labeled_mol)
        n_torsions += len(label_dict[f'{name}_torsion_inds'])
    summary = f'# atoms: {n_atoms}, # bonds: {n_bonds}, # angles: {n_angles}, # torsions: {n_torsions}'
    print(summary)

    # save to compressed array
    description = f"""
    Each of the molecules in AlkEthOH_rings.smi
        {alkethoh_url}
    
    is labeled according to the forcefield `openff_unconstrained-1.0.0.offxml`:
        https://github.com/openforcefield/openforcefields/blob/master/openforcefields/offxml/openff_unconstrained-1.0.0.offxml
    
    Keys are of the form 
        <name>_<atom|bond|angle|torsion>_<inds|labels>
        
        such as 'AlkEthOH_r0_atom_inds' or 'AlkEthOH_r0_torsion_labels'
        
    and values are integer arrays.
    
    {summary}
    """
    np.savez_compressed(path_to_npz,
                        description=description,
                        **label_dict)
