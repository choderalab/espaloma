import torch
from openforcefield.topology import Molecule

from espaloma.redux.energy import compute_valence_energy
from espaloma.redux.nn import TAG, MLP
from espaloma.redux.symmetry import ValenceModel, Readouts, elements

node_dim = 128
node_representation = TAG(in_dim=len(elements), hidden_dim=128, out_dim=node_dim)
readouts = Readouts(atoms=MLP(node_dim, 2), bonds=MLP(2 * node_dim, 2), angles=MLP(3 * node_dim, 2),
                    propers=MLP(4 * node_dim, 6), impropers=MLP(4 * node_dim, 6))
valence_model = ValenceModel(node_representation, readouts)

offmol = Molecule.from_smiles('CCC')
params = valence_model.forward(offmol)
print(params)

xyz = torch.randn((10, offmol.n_atoms, 3))
print(compute_valence_energy(offmol, xyz, params))
