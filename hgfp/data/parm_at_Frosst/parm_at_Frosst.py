"""Fetch molecules and assigned types from http://www.ccl.net/cca/data/parm_at_Frosst/ ,
define a generator that yields (openff_molecule, one_hot_atom_types) pairs."""

import tarfile
from os.path import exists

import numpy as np
from openforcefield.topology import Molecule

fname = 'parm_at_Frosst.tgz'
url = 'http://www.ccl.net/cca/data/parm_at_Frosst/parm_at_Frosst.tgz'

# download if we haven't already
if not exists(fname):
    print('Downloading {} from {}...'.format(fname, url))
    import urllib.request

    urllib.request.urlretrieve(url, fname)

# extract zinc and parm@frosst atom types
archive = tarfile.open(fname)

zinc_file = archive.extractfile('parm_at_Frosst/zinc.sdf')
zinc_p_f_types_file = archive.extractfile('parm_at_Frosst/zinc_p_f_types.txt')

zinc_p_f_types = [l.strip() for l in zinc_p_f_types_file.readlines()]
zinc_mols = Molecule.from_file(
    zinc_file,
    file_format='sdf',
    allow_undefined_stereo=True)

archive.close()

# convert types from strings to ints, for one-hot encoding
unique_types = sorted(list(set(zinc_p_f_types)))
n_types = len(unique_types)
type_to_int = dict(zip(unique_types, range(len(unique_types))))
type_ints = np.array([type_to_int[t] for t in zinc_p_f_types])


# define generators
def zinc_p_f_atom_types_generator():
    """generate (openforcefield.topology.Molecule, np.array) pairs"""
    current_index = 0
    for mol in zinc_mols:
        y = np.zeros((mol.n_atoms, n_types))
        for i in range(mol.n_atoms):
            y[i, type_ints[current_index]] = 1
            current_index += 1
        yield (mol, y)


# TODO: zinc_am1bcc_atom_types
# TODO: zinc_am1bcc_bond_types

# TODO: mmff94_p_f_atom_types
# TODO: mmff94_am1bcc_atom_types
# TODO: mmff94_am1bcc_bond_types


if __name__ == '__main__':
    from tqdm import tqdm

    print('sanity-checking shapes and row sums...')
    for (mol, y) in tqdm(zinc_p_f_atom_types_generator()):
        assert (y.shape == (mol.n_atoms, n_types))
        assert ((y.sum(1) == 1).all())
