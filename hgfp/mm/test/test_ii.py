import hgfp
import torch
import dgl
import numpy
import numpy.testing as npt
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
AllChem.EmbedMolecule(mol)
g = hgfp.graph.from_rdkit_mol(mol)

g = hgfp.heterograph.from_graph(g)

net = hgfp.models.gcn_with_combine_readout_ii.Net(
    [128, 'tanh', 128, 'tanh', 128, 'tanh'])

g = net(g, return_graph=True)


g = hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(g)

g = hgfp.mm.energy_in_heterograph_ii.u(g)

print(g.nodes['mol'])
