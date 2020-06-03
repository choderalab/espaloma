# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import torch
import dgl
import random
import espaloma as esp

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def from_csv(path, toolkit="rdkit", smiles_col=-1, y_cols=[-2], seed=2666):
    """ Read csv from file.
    """

    def _from_csv():
        df = pd.read_csv(path)
        df_smiles = df.iloc[:, smiles_col]
        df_y = df.iloc[:, y_cols]

        if toolkit == "rdkit":
            from rdkit import Chem

            mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]
            gs = [esp.HomogeneousGraph(mol) for mol in mols]

        elif toolkit == "openeye":
            from openeye import oechem

            mols = [
                oechem.OESmilesToMol(oechem.OEGraphMol(), smiles)
                for smiles in df_smiles
            ]
            gs = [esp.HomogeneousGraph(mol) for mol in mols]

        ds = list(zip(gs, list(torch.tensor(df_y.values))))
        random.seed(seed)
        random.shuffle(ds)

        return ds

    return _from_csv


def normalize(ds):
    """ Get mean and std.
    """

    gs, ys = tuple(zip(*ds))
    y_mean = np.mean(ys)
    y_std = np.std(ys)

    def norm(y):
        return (y - y_mean) / y_std

    def unnorm(y):
        return y * y_std + y_mean

    return y_mean, y_std, norm, unnorm


def split(ds, partition):
    """ Split the dataset according to some partition.
    """
    n_data = len(ds)

    # get the actual size of partition
    partition = [int(n_data * x / sum(partition)) for x in partition]

    ds_batched = []
    idx = 0
    for p_size in partition:
        ds_batched.append(ds[idx : idx + p_size])
        idx += p_size

    return ds_batched


def batch(ds, batch_size, seed=2666):
    """ Batch graphs and values after shuffling.
    """
    # get the numebr of data
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    random.seed(seed)
    random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        torch.stack(ys[idx * batch_size : (idx + 1) * batch_size], dim=0)
        for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def collate_fn(graphs):
    return dgl.batch(graphs)
