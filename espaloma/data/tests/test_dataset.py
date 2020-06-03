import pytest

@pytest.fixture
def smiles():
    import os
    import espaloma as esp
    import pandas as pd
    path = os.path.dirname(
            esp.__file__) + '/data/esol.csv'

    df = pd.read_csv(path)

    return df.iloc[:16, -1]
    
@pytest.fixture
def mols(smiles):
    print(smiles)

    import openforcefield
    from openforcefield.topology import Molecule
    return [Molecule.from_smiles(
        _smiles, allow_undefined_stereo=True) for _smiles in smiles]


def test_homo_ds(mols):
    import espaloma as esp
    ds = esp.data.dataset.HomogeneousGraphDataset(mols)

@pytest.fixture
def mol_ds(mols):
    import espaloma as esp
    ds = esp.data.dataset.MoleculeDataset(mols)
    return ds

def test_typing(mol_ds):
    homo_ds = mol_ds.apply_legacy_typing_homogeneous()
    next(iter(homo_ds))

def test_dataloader(mol_ds):
    import torch
    import dgl
    import espaloma as esp

    homo_ds = mol_ds.apply_legacy_typing_homogeneous()
    
    collate_fn = esp.data.utils.collate_fn 


    dataloader = torch.utils.data.DataLoader(
            homo_ds,
            collate_fn=collate_fn)

    print(next(iter(dataloader)))
