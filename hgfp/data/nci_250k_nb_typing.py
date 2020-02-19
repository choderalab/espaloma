# for each atom, get the binary vector of SMARTS matches that determine Parsley's nonbonded types
import re

import numpy as np
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from tqdm import tqdm

# load the parsley forcefield and extract the smirks used for vdW types
forcefield = ForceField('openff_unconstrained-1.0.0.offxml')


def extract_all_vdW_smirks(forcefield):
    ff_string = forcefield.to_string()
    start_ind = ff_string.find('<vdW')
    end_ind = ff_string.find('</vdW')

    prefix = 'smirks="'
    matches = re.findall(r'smirks=.*"', ff_string[start_ind:end_ind])
    return [m.split()[0][len(prefix):-1] for m in matches]


all_vdW_smirks = extract_all_vdW_smirks(forcefield)

# load the nci-250k smiles list from
# https://github.com/openforcefield/qca-dataset-submission/blob/2b1196fa723bb60bade9dc12792a8b515c006488/2019-07-05%20OpenFF%20NCI250K%20Boron%201/nci-250k.smi.gz
with open('nci-250k.smi', 'r') as f:
    smiles_list = [s.strip() for s in f.readlines()]


def get_vdW_match_matrix(mol):
    """return an (n_atoms, n_smirks) binary matrix"""
    n_atoms = mol.n_atoms
    n_smirks = len(all_vdW_smirks)
    match_matrix = np.zeros((n_atoms, n_smirks), dtype=bool)
    for j in range(len(all_vdW_smirks)):
        matches = mol.chemical_environment_matches(all_vdW_smirks[j])
        for (i,) in matches:
            match_matrix[i, j] = True
    return match_matrix


if __name__ == '__main__':
    batch_size = 5000
    np.random.seed(0)
    np.random.shuffle(smiles_list)

    n_batches = int(len(smiles_list) / batch_size)

    for batch in range(n_batches):
        targets = dict()
        for smiles in tqdm(smiles_list[(batch * batch_size):(batch + 1) * batch_size]):
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            targets[smiles] = get_vdW_match_matrix(mol)

        np.savez_compressed('nci_250k_parsley_vdw_matches_batch_{}_of_{}.npz'.format(batch, n_batches), **targets)
