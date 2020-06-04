"""provide a pytorch dataset views of the interaction typing dataset"""

from pickle import load
from typing import Tuple

import numpy as np
import torch
from openforcefield.topology import Molecule
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset

categorical_loss = CrossEntropyLoss()


class AlkEthOHDataset(Dataset):
    def __init__(self):
        with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
            self._mols = load(f)

        self._label_dict = np.load('AlkEthOH_rings.npz')
        self._mol_names = sorted(list(self._mols.keys()))

    def _get_inds(self, mol_name: str, type_name: str) -> Tensor:
        return tensor(self._label_dict[f'{mol_name}_{type_name}_inds'])

    def _get_labels(self, mol_name: str, type_name: str) -> Tensor:
        return tensor(self._label_dict[f'{mol_name}_{type_name}_labels'])

    def _get_all_unique_labels(self, type_name: str):
        all_labels = set()
        for mol_name in self._mol_names:
            new_labels = set(self._label_dict[f'{mol_name}_{type_name}_labels'])
            all_labels.update(new_labels)
        return sorted(list(all_labels))

    def _get_mol_inds_labels(self, index: int, type_name: str) -> Tuple[Molecule, Tensor, Tensor]:
        mol_name = self._mol_names[index]
        mol = self._mols[mol_name]
        inds = self._get_inds(mol_name, type_name)
        labels = self._get_labels(mol_name, type_name)
        return mol, inds, labels

    def loss(self, index: int, predictions: Tensor) -> Tensor:
        raise (NotImplementedError)


class AlkEthOHAtomTypesDataset(AlkEthOHDataset):
    def __init__(self):
        super(AlkEthOHAtomTypesDataset, self).__init__()
        self.type_name = 'atom'
        all_labels = self._get_all_unique_labels(self.type_name)

        self._label_mapping = dict(zip(all_labels, range(len(all_labels))))
        self.n_classes = len(self._label_mapping)

    def __getitem__(self, index) -> Tuple[Molecule, Tensor, Tensor]:
        mol, inds, _labels = self._get_mol_inds_labels(index, self.type_name)
        labels = tensor([self._label_mapping[int(i)] for i in _labels])
        return mol, inds, labels

    def loss(self, index: int, predictions: Tensor) -> Tensor:
        """cross entropy loss"""
        _, _, target = self[index]
        assert (predictions.shape == (len(target), self.n_classes))

        return categorical_loss(predictions, target)


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
    atom_type_dataset = AlkEthOHAtomTypesDataset()
    n_classes = atom_type_dataset.n_classes

    index = 0
    n_atoms = atom_type_dataset[index][0].n_atoms
    predictions = torch.randn(n_atoms, n_classes, requires_grad=True)
    loss = atom_type_dataset.loss(index, predictions)
    loss.backward()
    print(predictions.grad)

    # bonds
    print(AlkEthOHBondTypesDataset()[0])
    # angles
    print(AlkEthOHAngleTypesDataset()[0])
    # torsions
    print(AlkEthOHTorsionTypesDataset()[0])
