"""provide a pytorch dataset views of the interaction typing dataset"""

from pickle import load
from typing import Tuple

import numpy as np
from openforcefield.topology import Molecule
from torch import tensor, Tensor
from torch.utils.data.dataset import Dataset


class AlkEthOHDataset(Dataset):
    def __init__(self):
        with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
            self._mols = load(f)

        self._label_dict = np.load('AlkEthOH_rings.npz')

    def _get_inds(self, index: str, type_name: str) -> Tensor:
        return tensor(self._label_dict[f'{index}_{type_name}_inds'])

    def _get_labels(self, index: str, type_name:str) -> Tensor:
        return tensor(self._label_dict[f'{index}_{type_name}_labels'])

    def _get_mol_inds_labels(self, index: str, type_name: str) -> Tuple[Molecule, Tensor, Tensor]:
        assert (index in self._mols)
        mol = self._mols[index]
        inds = self._get_inds(index, type_name)
        labels = self._get_labels(index, type_name)
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

    index = 'AlkEthOH_r0'
    # atoms
    print(AlkEthOHAtomTypesDataset()[index])
    # bonds
    print(AlkEthOHBondTypesDataset()[index])
    # angles
    print(AlkEthOHAngleTypesDataset()[index])
    # torsions
    print(AlkEthOHTorsionTypesDataset()[index])
