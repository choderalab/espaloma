# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
from rdkit import Chem
import pandas as pd
import dgl
import torch
import os
import hgfp
import random
from openeye import oechem

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mean_and_std(csv_path='gdb9.sdf.csv'):
    df_csv = pd.read_csv(csv_path, index_col=0)
    df_u298 = df_csv['u298_atom']
    return df_u298.mean(), df_u298.std()

def unbatched(num=-1, sdf_path='gdb9.sdf', csv_path='gdb9.sdf.csv', hetero=True):
    # parse data
    df_csv = pd.read_csv(csv_path, index_col=0)
    # df_sdf = Chem.SDMolSupplier(sdf_path)
    
    ifs = oechem.oemolistream()
    ifs.open(sdf_path)
    df_sdf = ifs.GetOEGraphMols()

    # get u298 only
    df_u298 = df_csv['u298_atom']

    # init
    ds = []

    idx = 0

    while True:
        try:
            mol = next(df_sdf)
        except:
            break
        if num != -1 and idx > num:
            break
        if mol != None:
            n_atoms = mol.NumAtoms()
            if n_atoms > 2:
                # get the name of the molecule
                # name = mol.GetProp('_Name')
                
                try:
                    u = hgfp.data.mm_energy.u(mol, toolkit='openeye')
                except:
                    pass

                # get its u298
                u = torch.squeeze(torch.Tensor([u]))
                g = hgfp.graph.from_oemol(mol)

                if hetero is True:
                    g = hgfp.heterograph.from_graph(g)

                ds.append((g, u))


                idx += 1

    random.shuffle(ds)

    return lambda: iter(ds)


def batched(
        num=-1,
        sdf_path='gdb9.sdf',
        csv_path='gdb9.sdf.csv',
        n_batches_in_buffer=12,
        batch_size=32,
        cache=True,
        hetero=False,
        **kwargs):

    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num, sdf_path=sdf_path, csv_path=csv_path, hetero=hetero),
        n_batches_in_buffer=n_batches_in_buffer,
        batch_size=batch_size,
        cache=cache,
        hetero=hetero,
        **kwargs)
