import matplotlib.pyplot as plt
import torch
from openforcefield.topology import Molecule
from tqdm import tqdm

from espaloma.redux.energy import compute_valence_energy
from espaloma.redux.nn import TAG, MLP
from espaloma.redux.symmetry import ValenceModel, Readouts, elements


def initialize(hidden_dim=128, node_dim=128):
    node_representation = TAG(in_dim=len(elements), hidden_dim=hidden_dim, out_dim=node_dim)
    readouts = Readouts(atoms=MLP(node_dim, 2), bonds=MLP(2 * node_dim, 2), angles=MLP(3 * node_dim, 2),
                        propers=MLP(4 * node_dim, 6), impropers=MLP(4 * node_dim, 6))
    valence_model = ValenceModel(node_representation, readouts)
    return valence_model


# sanity-check: try to recover a valence model that we know to be exactly representable
torch.manual_seed(0)  # set random seed
ref_hyperparams = dict(hidden_dim=32, node_dim=32)
train_hyperparams = dict(hidden_dim=128, node_dim=128)  # bigger, guaranteed more expressive
ref_valence_model = initialize(**ref_hyperparams)
valence_model = initialize(**train_hyperparams)

caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
offmol = Molecule.from_smiles(caffeine_smiles)
ref_params = ref_valence_model.forward(offmol)

xyz = torch.randn((100, offmol.n_atoms, 3))
ref_energies = compute_valence_energy(offmol, xyz, ref_params)


def loss():
    params = valence_model.forward(offmol)
    energies = compute_valence_energy(offmol, xyz, params)
    squared_residuals = (energies - ref_energies) ** 2
    return torch.mean(squared_residuals)


# optimize using L-BFGS
lbfgs_options = dict(line_search_fn='strong_wolfe', history_size=10, max_iter=1)
# max_iter is # of L-BFGS iterations per torch.optim "step"

optimizer = torch.optim.LBFGS(valence_model.parameters(), **lbfgs_options)


def closure():
    optimizer.zero_grad()
    L = loss()
    L.backward(retain_graph=True)
    return L


loss_traj = [loss().detach().numpy()]
valence_model.train()

trange = tqdm(range(1000))
for epoch in trange:
    L = optimizer.step(closure).detach().numpy()

    loss_traj.append(float(L))
    trange.set_postfix(loss=L)

# plot result
plt.plot(loss_traj)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('# L-BFGS iterations')
plt.title('cloning a GNN-based reference valence model\nfrom snapshots and energies')
plt.ylabel('valence energy loss\n(MSE, in arbitrary energy unit^2)')
plt.tight_layout()
plt.savefig('recovering_GNN_valence_model.png', dpi=300, bbox_inches='tight')
plt.close()
