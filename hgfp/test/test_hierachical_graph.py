import numpy
import numpy.testing as npt
import hgfp
import rdkit
from rdkit import Chem


hg = hgfp.hierachical_graph.from_rdkit_mol(Chem.MolFromSmiles('Cc1ccccc1'))

print(hg)
