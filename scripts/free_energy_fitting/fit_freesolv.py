# TODO: Higher-quality vacuum trajectories
# TODO: Implement early stopping?
# TODO: Do many train/validation splits and report averages / stddevs
# TODO: Currently the input is just element identity, one-hot-encoded
#   --> switch to using espaloma's atom featurizer
# TODO: Update loss from un-scaled RMSE to something more like a likelihood
#   (incorporating expt error + free-energy-estimator error)

# TODO: update to use Espaloma atom features instead of one-hot element atom features

import matplotlib.pyplot as plt
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
import numpy as np

from espaloma.redux.nn import GraphSAGE, MLP
from espaloma.redux.symmetry import ValenceModel, Readouts, elements

from tqdm import tqdm

df = pd.read_hdf('freesolv_with_samples.h5')

from openmmtools.constants import kB
from simtk import unit
from espaloma.units import DISTANCE_UNIT, ENERGY_UNIT

temperature = 300 * unit.kelvin
kT = kB * temperature

# conversion from espaloma energy unit to kT
to_kT = 1.0 * ENERGY_UNIT / kT

from scipy.spatial.distance import pdist, squareform

# these are espaloma units of bohr
df['distance_matrices'] = None

# TODO: replace this with thorough vacuum sampling
trajectory_column = 'quick_xyz'
# trajectory_column = 'xyz'

for key in df.index:
    xyz = (df[trajectory_column][key] * unit.nanometer).value_in_unit(
        DISTANCE_UNIT)
    distance_matrices = [torch.tensor(squareform(pdist(conf))) for conf in xyz]
    distance_matrices = torch.stack(distance_matrices)
    df['distance_matrices'][key] = distance_matrices

from espaloma.mm.implicit import gbsa_obc2_energy

torch.manual_seed(12345)
np.random.seed(12345)

n_mols_per_batch = 10
n_snapshots_per_mol = 25
n_iterations = 1000


def compute_obc2_energies(
        distance_matrices,
        radii, scales, charges,
        alpha=0.8, beta=0.0, gamma=2.909125
):
    """Note: loops over all distance matrices in Python, incurring significant
    overhead
    TODO: should replace this with torchscript JIT version for speed
    """
    N = len(distance_matrices)
    E_s = torch.zeros(N)
    for i in range(N):
        E_s[i] += gbsa_obc2_energy(
            distance_matrices[i],
            radii, scales, charges,
            alpha, beta, gamma,
        )
    return E_s


def initialize(hidden_dim=128, node_dim=128, atom_dim=2):
    node_representation = GraphSAGE(in_dim=len(elements), hidden_dim=hidden_dim,
                              out_dim=node_dim)
    readouts = Readouts(atoms=MLP(node_dim, atom_dim),
                        bonds=MLP(2 * node_dim, 2),
                        angles=MLP(3 * node_dim, 2),
                        propers=MLP(4 * node_dim, 6),
                        impropers=MLP(4 * node_dim, 6))
    graph_model = ValenceModel(node_representation, readouts)
    return graph_model


def predict_obc2_params(offmol, graph_model):
    """output of graph net's atom representation will initially
    be near zero, just offset by a constant"""
    params = graph_model.forward(offmol)
    radii = params.atoms[:, 0] + 2
    scales = params.atoms[:, 1] + 1
    return radii, scales


def one_sided_exp(w):
    delta_f = - (torch.logsumexp(- w, dim=(0,)) - np.log(len(w)))
    return delta_f


def predict_on_key(key: str, graph_model, batch_size: int = 25) -> float:
    offmol = df['offmol'][key]

    radii, scales = predict_obc2_params(offmol, graph_model)

    distance_matrices = df['distance_matrices'][key]
    inds = np.random.randint(0, len(distance_matrices), size=batch_size)

    charges = torch.tensor(offmol.partial_charges / unit.elementary_charge)

    obc2_energies = compute_obc2_energies(distance_matrices[inds], radii,
                                          scales, charges)
    w = obc2_energies * to_kT
    pred_delta_f = one_sided_exp(w)

    return pred_delta_f


inds = list(df.index)
np.random.shuffle(inds)
train_inds = inds[::2]
valid_inds = inds[1::2]
print(f'len(train): {len(train_inds)}\nlen(valid): {len(valid_inds)}')

graph_model = initialize()

preds = []
for key in tqdm(train_inds):
    preds.append(predict_on_key(key, graph_model))

graph_model = initialize()

learning_rate = 1e-3
optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate)

# fit to a small chunk of the data
keys = train_inds[:n_mols_per_batch]

predict_dict = dict()
for key in tqdm(keys):
    predict_dict[key] = predict_on_key(key, graph_model)

initial_x = np.array(
    [predict_dict[key].detach() * kT / unit.kilocalorie_per_mole for key in
     keys])

predictions = []
batch_losses = []
trange = tqdm(range(n_iterations))

for t in trange:
    L = 0.0
    for key in keys:
        # make a free energy prediction using a random subset of snapshots
        prediction = predict_on_key(key, graph_model, n_snapshots_per_mol)
        target = (df['experimental value (kcal/mol)'][
                      key] * unit.kilocalorie_per_mole) / kT

        # TODO: modify loss function to depend on expt error and pred error
        L += (prediction - target) ** 2
        predictions.append((t, key, prediction))

    L /= len(keys)

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    batch_losses.append(L.detach().numpy())
    rmse_in_kcalmol = np.sqrt(
        batch_losses[-1] * kT / unit.kilocalories_per_mole)
    trange.set_postfix(rmse_in_kcalmol=rmse_in_kcalmol)

rmse_in_kcalmol = [np.sqrt(b * kT / unit.kilocalories_per_mole) for b in
                   batch_losses]

plt.figure(figsize=(6, 6))
title = f'GBSA hydration free energies\nFreeSolv subset overfitting check (n={len(initial_x)})'
plt.title(title)
plt.plot(rmse_in_kcalmol)
plt.ylim(0, )
plt.xlabel('Adam iterations')
plt.ylabel('RMSE loss (kcal/mol)')
plt.tight_layout()
plt.savefig('overfitting_rmse_traj.pdf')
plt.close()

final_predict_dict = dict()
for key in tqdm(keys):
    final_predict_dict[key] = predict_on_key(key, graph_model)

x = np.array(
    [final_predict_dict[key].detach() * kT / unit.kilocalorie_per_mole for key
     in keys])
y = [df['experimental value (kcal/mol)'][key] for key in keys]

plt.figure(figsize=(12, 6))

ax = plt.subplot(1, 2, 1)
plt.scatter(initial_x, y)
plt.plot([min(y), max(y)], [min(y), max(y)])
plt.xlabel('predicted (kcal/mol)')
plt.ylabel('reference (kcal/mol)')
plt.title(title + ' BEFORE training')

plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
plt.scatter(x, y)
plt.xlabel('predicted (kcal/mol)')
plt.ylabel('reference (kcal/mol)')
plt.title(title + ' AFTER training')
plt.plot([min(y), max(y)], [min(y), max(y)])

plt.tight_layout()

plt.savefig('freesolv_subset_overfitting.pdf')
plt.close()

# fit to a larger chunk of freesolv
graph_model = initialize()
optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate)

predictions = []
batch_losses = []
trange = tqdm(range(n_iterations))

for t in trange:

    # subsample the training set at each iteration...
    keys = np.random.choice(train_inds, size=n_mols_per_batch)

    L = 0.0
    for key in keys:
        # make a free energy prediction using a random subset of snapshots
        prediction = predict_on_key(key, graph_model, n_snapshots_per_mol)
        target = (df['experimental value (kcal/mol)'][
                      key] * unit.kilocalorie_per_mole) / kT

        # TODO: modify loss function to depend on expt error and pred error
        L += (prediction - target) ** 2
        predictions.append((t, key, prediction))

    L /= len(keys)

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    batch_losses.append(L.detach().numpy())
    rmse_in_kcalmol = np.sqrt(
        batch_losses[-1] * kT / unit.kilocalories_per_mole)
    trange.set_postfix(rmse_in_kcalmol=rmse_in_kcalmol)

title = f'GBSA hydration free energies\nFreeSolv 50:50 (n={len(train_inds)})'
rmse_in_kcalmol = [np.sqrt(b * kT / unit.kilocalories_per_mole) for b in
                   batch_losses]

# plot rmse loss trajectory
plt.figure(figsize=(6, 6))
plt.plot(rmse_in_kcalmol)
plt.xlabel('Adam iteration')
plt.ylabel('RMSE loss on current minibatch')
plt.ylim(0, )
plt.title(title + '\nminibatch training loss trajectory')
plt.tight_layout()
plt.savefig('freesolv_50_50_minibatch_loss_traj.pdf')
plt.close()

# plot final training / validation scatter plots
final_predict_dict = dict()
for key in tqdm(df.index):
    final_predict_dict[key] = predict_on_key(key, graph_model)

splits = {
    'training': train_inds,
    'validation': valid_inds,
}

plt.figure(figsize=(12, 6))
ax = None

for i, split_name in enumerate(splits):
    split = splits[split_name]
    x = np.array(
        [final_predict_dict[key].detach() * kT / unit.kilocalorie_per_mole for
         key in split])
    y = np.array([df['experimental value (kcal/mol)'][key] for key in split])

    rmse = np.sqrt(np.mean((x - y) ** 2))

    ax = plt.subplot(1, 2, i + 1, sharex=ax, sharey=ax)
    plt.scatter(x, y)
    plt.xlabel('predicted (kcal/mol)')
    plt.ylabel('reference (kcal/mol)')
    plt.plot([min(y), max(y)], [min(y), max(y)])

    plt.title(
        f'{title}\n({split_name} set: RMSE={rmse:.3f} kcal/mol)')
plt.tight_layout()
plt.savefig('freesolv_50_50_train_validate_scatter.pdf')
plt.close()

# compare with the RMSE that would be obtained by predicting a constant
constant_pred_baseline = np.sqrt(np.mean((y - np.mean(y)) ** 2))
print(
    f'RMSE that would be obtained by predicting a constant: {constant_pred_baseline:.3f} kcal/mol')
