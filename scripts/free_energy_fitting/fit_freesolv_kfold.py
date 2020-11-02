# TODO: Currently the input is just element identity, one-hot-encoded
#   --> switch to using espaloma's 117-dimensional atom featurizer
# TODO: Update loss from un-scaled RMSE to something more like a likelihood
#   (incorporating expt error + free-energy-estimator error)

from typing import Set, List

import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
import numpy as np

from espaloma.redux.nn import TAG, MLP
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

trajectory_column = 'xyz'

allowed_elements = {'C', 'H', 'O'}


def form_mini_freesolv(allowed_elements: Set[str]) -> List[str]:
    mini_freesolv = []
    for key in df.index:
        offmol = df['offmol'][key]
        if set([a.element.symbol for a in offmol.atoms]).issubset(
                allowed_elements):
            mini_freesolv.append(key)
    return mini_freesolv


mini_freesolv = form_mini_freesolv(allowed_elements)

for key in mini_freesolv:
    xyz = (df[trajectory_column][key] * unit.nanometer).value_in_unit(
        DISTANCE_UNIT)
    distance_matrices = [torch.tensor(squareform(pdist(conf))) for conf in xyz]
    distance_matrices = torch.stack(distance_matrices)
    df['distance_matrices'][key] = distance_matrices

from espaloma.mm.implicit import gbsa_obc2_energy


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
    node_representation = TAG(in_dim=len(elements), hidden_dim=hidden_dim,
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


def predict_on_key(key: str, graph_model, batch_size=25) -> float:
    offmol = df['offmol'][key]

    radii, scales = predict_obc2_params(offmol, graph_model)

    distance_matrices = df['distance_matrices'][key]
    if type(batch_size) == int:
        inds = np.random.randint(0, len(distance_matrices), size=batch_size)
    elif batch_size is None:
        inds = np.arange(np.arange(len(distance_matrices)))
    else:
        raise (RuntimeError('invalid batch_size argument'))

    charges = torch.tensor(offmol.partial_charges / unit.elementary_charge)

    obc2_energies = compute_obc2_energies(distance_matrices[inds], radii,
                                          scales, charges)
    w = obc2_energies * to_kT
    pred_delta_f = one_sided_exp(w)

    return pred_delta_f


def get_all_preds(keys, batch_size=50):
    predict_dict = dict()
    for key in keys:
        predict_dict[key] = float(predict_on_key(key, graph_model,
                                                 batch_size).detach() * kT / unit.kilocalorie_per_mole)
    return predict_dict


# one fold per processor
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
inds = sorted(list(mini_freesolv))

n_folds = 10
from sklearn.model_selection import KFold

kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
folds = []
for (train, valid) in kfold.split(inds):
    folds.append((train, valid))

if __name__ == '__main__':

    import sys

    fold_index = int(sys.argv[1])

    n_mols_per_batch = 10
    n_snapshots_per_mol = 25
    n_iterations = 10000
    learning_rate = 1e-3

    train, valid = folds[fold_index]

    train_inds = [inds[i] for i in train]
    valid_inds = [inds[i] for i in valid]
    print(f'len(train): {len(train_inds)}\nlen(valid): {len(valid_inds)}')


    def report_train_and_validation_rmse(predict_dict):
        train_residuals = np.array(
            [predict_dict[key] - df['experimental value (kcal/mol)'][key] for
             key in train_inds])
        validation_residuals = np.array(
            [predict_dict[key] - df['experimental value (kcal/mol)'][key] for
             key in valid_inds])

        return np.sqrt(np.mean(train_residuals ** 2)), np.sqrt(
            np.mean(validation_residuals ** 2))


    graph_model = initialize()

    optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate)

    predictions = []

    batch_losses = []
    trange = tqdm(range(n_iterations))
    n_batches_per_epoch = int(len(train_inds) / n_mols_per_batch)

    for t in trange:

        # optionally subsample the training set...
        keys = np.random.choice(train_inds, size=n_mols_per_batch)

        L = 0.0
        for key in keys:
            # make a free energy prediction using a random subset of snapshots for each key
            prediction = predict_on_key(key, graph_model, n_snapshots_per_mol)
            target = (df['experimental value (kcal/mol)'][
                          key] * unit.kilocalorie_per_mole) / kT

            # TODO: modify loss function to depend on experimental error and simulation error
            L += (prediction - target) ** 2

    L /= len(keys)

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    batch_losses.append(L.detach().numpy())
    rmse_in_kcalmol = np.sqrt(
        batch_losses[-1] * kT / unit.kilocalories_per_mole)
    trange.set_postfix(batch_rmse_in_kcalmol=rmse_in_kcalmol)

    if t % n_batches_per_epoch == 0:
        epoch = int(t / n_batches_per_epoch)
        p = get_all_preds(mini_freesolv)
        predictions.append(p)
        train_rmse, valid_rmse = report_train_and_validation_rmse(p)
        print(f'training rmse: {train_rmse:.3f}')
        print(f'validation rmse: {valid_rmse:.3f}')
        name = f'cho_freesolv_fold={fold_index}_epoch={epoch}'
        if (t % n_batches_per_epoch) / 10 == 0:
            torch.save(graph_model, f'{name}.pt')

        # TODO: make this a nice pandas dataframe instead
        from pickle import dump

        with open(f'{name}.pkl', 'wb') as f:
            dump({
                'predictions': predictions,
                'train_inds': train_inds,
                'valid_inds': valid_inds},
                f)
