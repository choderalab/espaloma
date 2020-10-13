import torch
from openforcefield.topology import Molecule

from espaloma.redux.energy import compute_valence_energy
from espaloma.redux.nn import TAG, MLP
from espaloma.redux.symmetry import ValenceModel, Readouts, elements

def initialize(hidden_dim=128, node_dim=128):
    node_representation = TAG(in_dim=len(elements), hidden_dim=hidden_dim, out_dim=node_dim)
    readouts = Readouts(atoms=MLP(node_dim, 2), bonds=MLP(2 * node_dim, 2), angles=MLP(3 * node_dim, 2),
                        propers=MLP(4 * node_dim, 6), impropers=MLP(4 * node_dim, 6))
    valence_model = ValenceModel(node_representation, readouts)
    return valence_model

ref_valence_model = initialize()
valence_model = initialize()

offmol = Molecule.from_smiles('CCC')
ref_params = ref_valence_model.forward(offmol)
print(ref_params)

xyz = torch.randn((10, offmol.n_atoms, 3))
ref_energies = compute_valence_energy(offmol, xyz, ref_params)
print(ref_energies)


def loss():
    params = valence_model.forward(offmol)
    energies = compute_valence_energy(offmol, xyz, params)
    return ((energies - ref_energies)**2).sum()

L = loss()
print(L)
L.backward()
print(valence_model.parameters())
