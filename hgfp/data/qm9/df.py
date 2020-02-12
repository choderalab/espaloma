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


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def unbatched(num=-1, sdf_path='../data/qm9/gdb9.sdf', csv_path='../data/qm9/gdb9.sdf.csv'):
    """ Put qm9 molecules in a dataset.
    """
    # parse data
    df_csv = pd.read_csv(csv_path, index_col=0)
    df_sdf = Chem.SDMolSupplier(sdf_path)

    # get u298 only
    df_u298 = df_csv['u298_atom']

    # initialize graph list to be empty

    def qm9_iter():
        idx = 0
        while True:
            mol = next(df_sdf)
            if num != -1 and idx > num:
                break
            if mol != None:
                n_atoms = mol.GetNumAtoms()
                if n_atoms > 2:
                    # get the name of the molecule
                    name = mol.GetProp('_Name')

                    # get its u298
                    u = torch.squeeze(torch.Tensor([df_u298[name]]))

                    # put molecule in graph list
                    g = dgl.DGLGraph()
                    g.add_nodes(n_atoms)
                    g.ndata['atoms'] = torch.Tensor(
                        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
                    conformer = mol.GetConformer()
                    g.ndata['xyz'] = torch.Tensor(
                        [
                            [
                                conformer.GetAtomPosition(idx).x,
                                conformer.GetAtomPosition(idx).y,
                                conformer.GetAtomPosition(idx).z
                            ] for idx in range(n_atoms)
                        ])

                    bonds = list(mol.GetBonds())
                    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
                    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
                    bonds_types = [bond.GetBondType().real for bond in bonds]

                    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
                    g.edata['type'] = torch.Tensor(bonds_types)[:, None]

                    idx += 1
                    yield(g, u)

    return qm9_iter

def batched(
        num=-1,
        sdf_path='../data/qm9/gdb9.sdf',
        csv_path='../data/qm9/gdb9.sdf.csv',
        n_batches_in_buffer=12,
        batch_size=32,
        cache=True):
    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num, sdf_path=sdf_path, csv_path=csv_path),
        n_batches_in_buffer=n_batches_in_buffer,
        batch_size=batch_size,
        cache=cache)
