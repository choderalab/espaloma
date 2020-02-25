"""
Entities:
* atom
* bond
* angle
* unoriented_plane
* torsion
* ring


Connection types:
* atom in bond
* atom in unoriented_plane
* atom in ring

* bond in angle
* bond in unoriented plane ?
* bond in ring

* unoriented_plane in torsion
"""

import dgl
from openforcefield.topology import Molecule

# TODO: atom in bond

# TODO: atom in unoriented plane

# TODO: bond in angle

# TODO: bond in ring?

# TODO: make the message-passing functions decision-trees or decision-DAGs
#   rather than neural nets
import numpy as np

atom_feature_dimension = 8

bond_feature_dimension = 3

ring_feature_dimension = 2

## GET FEATURE VECTORS FOR EACH OF THE ENTITIES ##

# atoms
def get_info_about_one_atom_rd(rd_atom):
    atom_info_tuple = (
        rd_atom.GetAtomicNum(), # int
        rd_atom.GetFormalCharge(), # int
        rd_atom.GetTotalDegree(), # int
        # GetHvyDegree()?
        rd_atom.GetHybridization().numerator, # get int in [0,1, ... 7]
        rd_atom.GetTotalNumHs(), # int
        rd_atom.GetTotalValence(), # int
        rd_atom.GetIsAromatic(), # bool
        rd_atom.IsInRing()  # bool
    )
    return atom_info_tuple


def get_atom_info_rd(rdmol):
    return np.array(list(map(get_info_about_one_atom_rd, rdmol.GetAtoms())))

# TODO bonds

def get_info_about_one_bond_rd(b):
    bond_info_tuple = (
        b.GetBondTypeAsDouble(),
        b.GetIsAromatic(),
        b.GetIsConjugated()
    )
    return bond_info_tuple

def get_bond_info_rd(rdmol):
    bonds = list(rdmol.GetBonds())
    return np.array(list(map(get_info_about_one_bond_rd, bonds)))


# rings



# rdkit to extract rings
def get_ring_memberships_rd(rdmol):
    ring_info = rdmol.GetRingInfo()
    # TODO: look into difference between AtomRings and BondRings?
    return ring_info.AtomRings()



def get_ring_info_rd(rdmol):
    # TODO: should I say a ring is aromatic only if all of the atoms involved are aromatic?
    atom_rings = get_ring_memberships_rd(rdmol)
    aromatic_atoms = set(rdmol.GetAromaticAtoms())

    ring_info = np.zeros((len(atom_rings), ring_feature_dimension))
    ring_info[:,1] = 1

    for i, ring in enumerate(atom_rings):
        ring_info[i,0] = len(ring)
        for atom in ring:
            if not atom in aromatic_atoms:
            #if not atom.GetIsAromatic():
                ring_info[i, 1] = 0
    return ring_info



# TODO: angles
# TODO: torsions

## GET EDGES FOR EACH OF THE ENTITIES

def get_atom_bond_edges(mol):
    bonds = list(mol.bonds)

    atom_in_bond_type = ('atom', 'atom_in_bond', 'bond')
    atom_in_bond_edges = []
    for i, bond in enumerate(bonds):
        atom_in_bond_edges.append((bond.atom1_index, i))
        atom_in_bond_edges.append((bond.atom2_index, i))
    return {atom_in_bond_type: atom_in_bond_edges}


def get_atom_ring_edges(rdmol):
    rings = get_ring_memberships_rd(rdmol)
    atom_in_ring_edges = []
    atom_in_ring_type = ('atom', 'atom_in_ring', 'ring')  # TODO: question: can I use "in" multiple times? seems to throw error, but annoying to be redundant...
    for i, ring in enumerate(rings):
        for atom in ring:
            atom_in_ring_edges.append((atom, i))
    # TODO: initialize data sitting on each ring!
    #   aromatic?
    #   number of members?

    return {atom_in_ring_type : atom_in_ring_edges}

def form_heterograph(mol):
    # TODO: allow constructor to use a subset of {atom, bond, ring, angle, torsion, ...}
    #   entities

    # TODO: add also edges in the opposite direction
    rdmol = mol.to_rdkit()
    all_edges = {}
    all_edges.update(get_atom_bond_edges(mol))
    all_edges.update(get_atom_ring_edges(rdmol))
    return dgl.heterograph(all_edges)


# TODO: form atom classification task

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size_dict, out_size_dict, canonical_etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                etype : nn.Linear(in_size_dict[srctype], out_size_dict[dsttype]) for (srctype, etype, dsttype) in canonical_etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCN(nn.Module):
    def __init__(self, canonical_etypes, in_size_dict, hidden_size_dict, out_size_dict):
        # TODO: problem, assumes a single in_size for all node types...
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.

        # create layers
        self.layer1 = HeteroRGCNLayer(in_size_dict, hidden_size_dict, canonical_etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size_dict, out_size_dict, canonical_etypes)

    def forward(self, G, feature_dict):
        h_dict = self.layer1(G, feature_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict['atom']


def form_feature_dict(mol):
    rdmol = mol.to_rdkit()
    return {
        'atom': get_atom_info_rd(rdmol),
        'bond': get_bond_info_rd(rdmol),
        'ring': get_ring_info_rd(rdmol)
    }

from pickle import load
if __name__ == '__main__':
    smiles = 'c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45'
    mol = Molecule.from_smiles(smiles)
    print(mol)



    path = '/Users/joshuafass/Documents/GitHub/hgfp/hgfp/data/parm_at_Frosst/p_f_zinc_mols_and_targets.pkl'
    with open(path, 'rb') as f:
        holy_moly_s = load(f)[:100]

    from tqdm import tqdm

    heterographs = [form_heterograph(mol) for (mol, y) in tqdm(holy_moly_s)]
    feature_dicts = [form_feature_dict(mol) for (mol, y) in tqdm(holy_moly_s)]
    ys = [y for (mol, y) in holy_moly_s]  # one-hot targets...

    target_dim = ys[0].shape[1]
    hidden_dim = 100
    in_size_dict = {name:feature_dicts[0][name].shape[1] for name in feature_dicts[0]}

    hidden_size_dict = {name: hidden_dim for name in feature_dicts[0]}
    # TODO: for now ignoring the other types, going to output the same size on all entities
    out_size_dict = {name: target_dim for name in feature_dicts[0]}

    hg = form_heterograph(mol)
    print(hg)
    print(hg.etypes)



    model = HeteroRGCN(hg.canonical_etypes, in_size_dict, hidden_size_dict, out_size_dict)
    predictions = model.forward(heterographs[0], feature_dicts[0])








