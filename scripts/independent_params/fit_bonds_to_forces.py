# separate parameters for every atom, bond, angle, torsion, up to symmetry

from time import time

import matplotlib.pyplot as plt
import numpy as onp
from jax.config import config  # TODO: is there a way to enable this globally?

from scripts.independent_params.plots import plot_residuals, plot_loss_traj

config.update("jax_enable_x64", True)

from espaloma.utils.jax import jax_play_nice_with_scipy
from jax import grad, jit, numpy as np
from scipy.optimize import basinhopping

from espaloma.data.alkethoh.ani import get_snapshots_energies_and_forces
from espaloma.data.alkethoh.data import offmols
from espaloma.mm.mm_utils import get_force_targets, MMComponents, initialize_bonds
from espaloma.mm.mm_utils import compute_harmonic_bond_potential, get_sim
from espaloma.utils.symmetry import get_unique_bonds

onp.random.seed(1234)

# TODO: add coupling terms
# TODO: toggle between `compute_harmonic_bond_potential` and alternate parameterization

# TODO: initializer classes
#   initialize at mean values vs. at openff values


if __name__ == '__main__':
    # look at a single molecule first
    name = 'AlkEthOH_r4'
    offmol = offmols[name]
    # params, unpack = initialize_off_params(offmol)
    sim = get_sim(name)
    params = initialize_bonds(sim, offmol, noise_magnitude=1.0)
    pair_inds, bond_inds = get_unique_bonds(offmol)

    # targets
    mm_components, ani1ccx_forces = get_force_targets(name, MMComponents(bonds=True))
    target = mm_components.bonds

    # trajectory
    traj, _, _ = get_snapshots_energies_and_forces(name)
    xyz = traj.xyz


    @jit
    def compute_harmonic_bond_force(xyz, bond_params, pair_inds, bond_inds):
        total_U = lambda xyz: np.sum(compute_harmonic_bond_potential(xyz, bond_params, pair_inds, bond_inds))
        return - grad(total_U)(xyz)


    @jit
    def predict(all_params):
        bond_params = all_params
        F_bond = compute_harmonic_bond_force(xyz, bond_params, pair_inds, bond_inds)
        return F_bond


    def stddev_loss(predicted, actual):
        return np.std(predicted - actual)


    def rmse_loss(predicted, actual):
        return np.sqrt(np.mean((predicted - actual) ** 2))


    @jit
    def loss(all_params):
        """choices available here:
            * std vs. rmse loss
            * regularization vs. no regularization
            * different normalizations and scalings
        """
        return rmse_loss(predict(all_params), target)


    print('loss at initialization: {:.3f}'.format(loss(params)))

    # check timing
    g = grad(loss)
    t0 = time()
    _ = g(params)
    t1 = time()
    _ = g(params)
    t2 = time()
    print(f'time to compile gradient: {t1 - t0:.3f}s')
    print(f'time to compute gradient: {t2 - t1:.3f}s')

    # TODO: an optimization result class...
    # optimize, storing some stuff
    traj = [params]


    def callback(x):
        global traj
        traj.append(x)

    stop_thresh = 1e-3
    def bh_callback(x, bh_e=None, bh_accept=None):
        L = loss(x)
        print('loss: {:.5f}'.format(L))
        if L <= stop_thresh:
            print('stopping threshold reached ({:.5f} <= {:.5f}), terminating early'.format(L ,stop_thresh))
            return True


    method = 'BFGS'

    fun, jac = jax_play_nice_with_scipy(loss)
    min_options = dict(disp=True, maxiter=500)
    # opt_result = minimize(fun, x0=params, jac=jac, method=method,
    #                      options=min_options, callback=callback)

    # fictitious "temperature" -- from scipy.optimize.basinhopping documentation:
    #   The “temperature” parameter for the accept or reject criterion.
    #   Higher “temperatures” mean that larger jumps in function value will be accepted.
    #   For best results T should be comparable to the separation (in function value) between local minima.
    bh_temperature = 1.0

    opt_result = basinhopping(fun, params, T=bh_temperature,
                              minimizer_kwargs=dict(method=method, jac=jac, callback=callback, options=min_options),
                              callback=bh_callback)

    target_name = 'bond'
    plot_residuals(predict(opt_result.x), target, name, target_name)

    loss_traj = [fun(theta) for theta in traj]
    plot_loss_traj(loss_traj, method=method, mol_name=name, target_name=target_name)

    # now try with Langevin
    from espaloma.utils.samplers import langevin

    logprobfun = lambda x: - fun(x)
    gradlogprobfun = lambda x: - jac(x)
    x0 = opt_result.x
    v0 = onp.random.randn(*x0.shape)
    langevin_traj, langevin_log_prob_traj = langevin(x0, v0, logprobfun, gradlogprobfun, stepsize=1e-4, collision_rate=np.inf, n_steps=10000)


    plt.plot(-langevin_log_prob_traj)
    plt.xlabel('langevin iteration')
    plt.ylabel('loss')
    plt.title(f'initialized from {method} result')
    plt.savefig(f'plots/{name}_langevin_loss_traj.png', dpi=300)
    plt.close()

    # initialize from random
    x0 = params
    v0 = onp.random.randn(*x0.shape)
    langevin_traj, langevin_log_prob_traj = langevin(x0, v0, logprobfun, gradlogprobfun, stepsize=1e-4,
                                                     collision_rate=np.inf, n_steps=10000)

    plt.plot(-langevin_log_prob_traj)
    plt.xlabel('langevin iteration')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title(f'initialized from random')
    plt.savefig(f'plots/{name}_langevin_loss_traj_random_init.png', dpi=300)
    plt.close()
