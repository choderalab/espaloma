import numpy
import numpy.testing as npt
import hgfp
import rdkit
from rdkit import Chem

g = hgfp.graph.from_rdkit_mol(Chem.MolFromSmiles('Cc1ccccc1'))

hg = hgfp.heterograph.from_graph(g)

print(hg)
