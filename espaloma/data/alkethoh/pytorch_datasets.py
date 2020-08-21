"""provide a pytorch dataset views of the interaction typing dataset"""

from pickle import load
from typing import Tuple

import numpy as np
from openforcefield.topology import Molecule
from torch import tensor, Tensor
from torch.utils.data.dataset import Dataset

from pkg_resources import resource_filename
path_to_offmols = resource_filename('espaloma.data.alkethoh', 'AlkEthOH_rings_offmols.pkl')
path_to_npz = resource_filename('espaloma.data.alkethoh', 'AlkEthOH_rings.npz')

class AlkEthOHDataset(Dataset):
    def __init__(self):
        with open(path_to_offmols, 'rb') as f:
            self._mols = load(f)

        self._label_dict = np.load(path_to_npz)
        self._mol_names = sorted(list(self._mols.keys()))

    def _get_inds(self, mol_name: str, type_name: str) -> Tensor:
        return tensor(self._label_dict[f'{mol_name}_{type_name}_inds'])

    def _get_labels(self, mol_name: str, type_name: str) -> Tensor:
        return tensor(self._label_dict[f'{mol_name}_{type_name}_labels'])

    def _get_mol_inds_labels(self, index: int, type_name: str) -> Tuple[Molecule, Tensor, Tensor]:
        mol_name = self._mol_names[index]
        mol = self._mols[mol_name]
        inds = self._get_inds(mol_name, type_name)
        labels = self._get_labels(mol_name, type_name)
        return mol, inds, labels


class AlkEthOHAtomTypesDataset(AlkEthOHDataset):

    def __getitem__(self, index) -> Tuple[Molecule, Tensor, Tensor]:
        return self._get_mol_inds_labels(index, 'atom')


class AlkEthOHBondTypesDataset(AlkEthOHDataset):

    def __getitem__(self, index) -> Tuple[Molecule, Tensor, Tensor]:
        return self._get_mol_inds_labels(index, 'bond')


class AlkEthOHAngleTypesDataset(AlkEthOHDataset):

    def __getitem__(self, index) -> Tuple[Molecule, Tensor, Tensor]:
        return self._get_mol_inds_labels(index, 'angle')


class AlkEthOHTorsionTypesDataset(AlkEthOHDataset):

    def __getitem__(self, index) -> Tuple[Molecule, Tensor, Tensor]:
        return self._get_mol_inds_labels(index, 'torsion')


if __name__ == '__main__':
    # TODO: move this from __main__ into doctests

    # atoms
    print(AlkEthOHAtomTypesDataset()[0])
    # bonds
    print(AlkEthOHBondTypesDataset()[0])
    # angles
    print(AlkEthOHAngleTypesDataset()[0])
    # torsions
    print(AlkEthOHTorsionTypesDataset()[0])
