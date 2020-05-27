"""label every molecule in AlkEthOH rings set"""

import os
import urllib

import numpy as np
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from tqdm import tqdm

alkethoh_url = 'https://raw.githubusercontent.com/openforcefield/open-forcefield-data/e07bde16c34a3fa1d73ab72e2b8aeab7cd6524df/Model-Systems/AlkEthOH_distrib/AlkEthOH_rings.smi'
alkethoh_local_path = 'AlkEthOH_rings.smi'


def download_alkethoh():
    if not os.path.exists(alkethoh_local_path):
        with urllib.request.urlopen(alkethoh_url) as response:
            smi = response.read()
        with open(alkethoh_local_path, 'wb') as f:
            f.write(smi)


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
get_labeled_torsions = partial(get_inds_and_labels, type_name='Torsions')

if __name__ == '__main__':
    # download, if it isn't already present
    download_alkethoh()

    # load molecules
    with open(alkethoh_local_path, 'r') as f:
        ring_smiles = [l.split()[0] for l in f.readlines()]
    with open(alkethoh_local_path, 'r') as f:
        ring_names = [l.split()[1] for l in f.readlines()]
    mols = [Molecule.from_smiles(s, allow_undefined_stereo=True) for s in ring_smiles]

    # Label molecules using forcefield
    # Takes about ~200ms per molecule -- can do ~1000 molecules in ~5-6 minutes, sequentially
    labeled_mols = []
    for mol in tqdm(mols):
        labeled_mols.append(label_mol(mol))

    label_dict = dict()
    for (name, labeled_mol) in zip(ring_names, labeled_mols):
        label_dict[name + '_atom_inds'], label_dict[name + '_atom_labels'] = get_labeled_atoms(labeled_mol)
        label_dict[name + '_bond_inds'], label_dict[name + '_bond_labels'] = get_labeled_bonds(labeled_mol)
        label_dict[name + '_angle_inds'], label_dict[name + '_angle_labels'] = get_labeled_angles(labeled_mol)
        label_dict[name + '_torsion_inds'], label_dict[name + '_torsion_labels'] = get_labeled_torsions(labeled_mol)

    # save to compressed array
    description = f"""
    Each of the molecules in AlkEthOH_rings.smi
        {alkethoh_url}
    
    is labeled according to the forcefield `openff_unconstrained-1.0.0.offxml`:
        https://github.com/openforcefield/openforcefields/blob/master/openforcefields/offxml/openff_unconstrained-1.0.0.offxml
    
    Keys are of the form 
        <name>_<atom|bond|angle|torsion>_<inds|labels>
        
        such as 'AlkEthOH_r0_atom_inds' or 'AlkEthOH_r0_torsion_labels'
        
    and values are integer arrays
    """

    np.savez_compressed('AlkEthOH_rings.npz',
                        description=description,
                        **label_dict)
