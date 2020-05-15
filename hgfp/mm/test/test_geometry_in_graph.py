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

hg = hgfp.heterograph.from_graph(g)

print(hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(hg))
