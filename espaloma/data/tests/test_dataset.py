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

@pytest.fixture
def homo_ds(mol_ds):
    return mol_ds.apply_legacy_typing_homogeneous()

def test_dataloader(homo_ds):
    import torch
    import dgl
    import espaloma as esp

    collate_fn = esp.data.utils.collate_fn 

    dataloader = torch.utils.data.DataLoader(
            homo_ds,
            collate_fn=collate_fn)

def test_save_load_homo(homo_ds):
    import tempfile
    import espaloma as esp
    import torch
    with tempfile.TemporaryDirectory() as tempdir:
        homo_ds.save(tempdir + '/ds.esp')
        new_homo_ds = esp.data.dataset.HomogeneousGraphDataset()
        new_homo_ds.load(tempdir + '/ds.esp')
        
        for old_graph, new_graph in zip(
                iter(homo_ds), iter(new_homo_ds)):
            assert old_graph.number_of_nodes() == new_graph.number_of_nodes()
            assert old_graph.number_of_edges() == new_graph.number_of_edges()
            assert torch.equal(
                    old_graph.ndata['h0'], 
                    new_graph.ndata['h0'])

