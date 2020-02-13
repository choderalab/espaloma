import dgl
import torch

def from_rdkit_mol(mol):
    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.GetNumAtoms()
    g.add_nodes(n_atoms)
    g.ndata['type'] = torch.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()])

    try:
        # enter xyz in if there is conformer
        conformer = mol.GetConformer()
        g.ndata['xyz'] = torch.Tensor(
            [
                [
                    conformer.GetAtomPosition(idx).x,
                    conformer.GetAtomPosition(idx).y,
                    conformer.GetAtomPosition(idx).z
                ] for idx in range(n_atoms)
            ])
    except:
        pass

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    bonds_types = [bond.GetBondType().real for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    g.edata['type'] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g
