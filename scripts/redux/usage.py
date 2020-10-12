from nn import TAG, MLP
from symmetry import ValenceModel, Readouts, elements

from openforcefield.topology import Molecule
terms = ['atoms', 'bonds', 'angles', 'propers', 'impropers']

node_dim = 128
node_representation = TAG(in_dim=len(elements), hidden_dim=128, out_dim=node_dim)
readouts = Readouts(atoms=MLP(node_dim, 2), bonds=MLP(2 * node_dim, 2), angles=MLP(3 * node_dim, 2),
                    propers=MLP(4 * node_dim, 6), impropers=MLP(4 * node_dim, 6))


offmol = Molecule.from_smiles('CCC')
valence_model = ValenceModel(node_representation, readouts)
print(valence_model.forward(offmol))