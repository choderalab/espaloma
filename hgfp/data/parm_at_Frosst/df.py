"""Fetch molecules and assigned types from http://www.ccl.net/cca/data/parm_at_Frosst/ ,
define a generator that yields (openff_molecule, one_hot_atom_types) pairs."""

import tarfile
from os.path import exists
import logging
logger = logging.getLogger()
from rdkit import Chem
import hgfp

import numpy as np
from openforcefield.topology import Molecule
import torch

def unbatched(num=-1, use_fp=True):
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

    zinc_mols = Chem.ForwardSDMolSupplier(zinc_file, removeHs=False)

    # archive.close()

    # convert types from strings to ints, for one-hot encoding
    unique_types = sorted(list(set(zinc_p_f_types)))
    n_types = len(unique_types)
    type_to_int = dict(zip(unique_types, range(len(unique_types))))
    int_to_type = dict(zip(range(len(unique_types)), unique_types))
    type_ints = np.array([type_to_int[t] for t in zinc_p_f_types])


    # define generators
    def _iter():
        """generate (openforcefield.topology.Molecule, np.array) pairs"""

        current_index = 0
        idx = 0
        for mol in zinc_mols:
            if num != -1 and idx > num:
                break

            y = np.zeros((mol.GetNumAtoms(), n_types))
            for i in range(mol.GetNumAtoms()):

                assert(int_to_type[type_ints[current_index]].decode('utf-8').startswith(str(mol.GetAtomWithIdx(i).GetSymbol())[0]))

                y[i, type_ints[current_index]] = 1
                current_index += 1
            #
            # assert (y.shape == (mol.GetNumAtoms(), n_types))
            # assert ((y.sum(1) == 1).all())
            idx += 1
            yield (hgfp.hierachical_graph.from_rdkit_mol(mol),
                torch.Tensor(y))


    return _iter

def batched(
        num=-1,
        batch_size=32,
        cache=True,
        use_fp=True):

    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num, use_fp=use_fp),
        batch_size=batch_size,
        cache=cache,
        hetero=True,
        cat_not_stack=True,
        n_batches_in_buffer=1)
