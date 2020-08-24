# separate parameters for every atom, angle, angle, torsion, up to symmetry

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
from espaloma.mm.mm_utils import get_force_targets, MMComponents, initialize_torsions
from espaloma.mm.mm_utils import compute_harmonic_bond_potential_alt_param, compute_harmonic_angle_potential_alt_param, compute_periodic_torsion_potential, get_sim
from espaloma.utils.symmetry import get_unique_bonds, get_unique_angles, get_unique_torsions

onp.random.seed(1234)

# TODO: add coupling terms
# TODO: toggle between `compute_harmonic_angle_potential` and alternate parameterization

# TODO: initializer classes
#   initialize at mean values vs. at openff values


if __name__ == '__main__':
    # look at a single molecule first
    name = 'AlkEthOH_r4'
    offmol = offmols[name]
    sim = get_sim(name)

    # TODO: also toggle whether this is log-scaled or not

    pair_inds, bond_inds = get_unique_bonds(offmol)
    triple_inds, angle_inds = get_unique_angles(offmol)
    quad_inds, torsion_inds = get_unique_torsions(offmol)

    # bonds
    n_unique_bonds = len(set(bond_inds))
    bond_params = np.ones(
        n_unique_bonds * 2) * 100000 / 2  # eyeballed, bonds are something like k=10000 kJ/mol / nm^2, r0=1 Å

    #angles
    n_unique_angles = len(set(angle_inds))
    angle_params = np.ones(n_unique_angles * 2) * 400 / 2 # eyeballed, angles are something like kJ/(mol rad**2)

    # torsions
    torsion_params = initialize_torsions(sim, offmol) # initialized with all zeros

    def pack(param_tuple):
        lengths = list(map(len, param_tuple))
        inds = [0] + list(onp.cumsum(lengths))
        def unpack(flat_params):
            return tuple([flat_params[inds[i]:inds[i+1]] for i in range(len(inds) - 1)])

        return np.hstack(param_tuple), unpack

    params, unpack = pack((bond_params, angle_params, torsion_params))


    # targets
    target_name = 'valence'
    mm_components, ani1ccx_forces = get_force_targets(name, MMComponents(valence=True))
    thinning = 10 # TODO: revert to whole thing, just reducing size to make this run faster
    target = mm_components.valence[::thinning] # TODO: revert to whole thing, just reducing size to make this run faster

    # trajectory
    traj, _, _ = get_snapshots_energies_and_forces(name)
    xyz = traj.xyz[::thinning] # TODO: revert to whole thing, just reducing size to make this run faster


    @jit
    def compute_harmonic_bond_force(xyz, bond_params, pair_inds, bond_inds):
        total_U = lambda xyz: np.sum(compute_harmonic_bond_potential_alt_param(xyz, bond_params, pair_inds, bond_inds))
        return - grad(total_U)(xyz)


    @jit
    def compute_harmonic_angle_force(xyz, angle_params, triple_inds, angle_inds):
        total_U = lambda xyz: np.sum(compute_harmonic_angle_potential_alt_param(xyz, angle_params, triple_inds, angle_inds))
        return - grad(total_U)(xyz)

    @jit
    def compute_torsion_force(xyz, torsion_params, quad_inds, torsion_inds):
        total_U = lambda xyz: np.sum(compute_periodic_torsion_potential(xyz, torsion_params, quad_inds, torsion_inds))
        return - grad(total_U)(xyz)


    @jit
    def predict(all_params):
        bond_params, angle_params, torsion_params = unpack(all_params)

        F_bond = compute_harmonic_bond_force(xyz, bond_params, pair_inds, bond_inds)
        F_angle = compute_harmonic_angle_force(xyz, angle_params, triple_inds, angle_inds)
        F_torsion = compute_torsion_force(xyz, torsion_params, quad_inds, torsion_inds)
        return F_bond + F_angle + F_torsion


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

    class MultiplicativeNoiseStep():
        def __init__(self, noise_magnitude=0.1):
            self.stepsize = noise_magnitude

        def __call__(self, x):
            # uniform between [1-self.stepsize, 1+self.stepsize]
            multiplicative_noise = 2 * self.stepsize * (onp.random.rand(*x.shape) - 0.5) + 1.0
            return x * multiplicative_noise

    method = 'BFGS'

    fun, jac = jax_play_nice_with_scipy(loss)
    min_options = dict(maxiter=500)
    from scipy.optimize import minimize
    opt_result = minimize(fun, x0=params, jac=jac, method=method,
                         options=min_options, callback=callback)
    print(f'loss after {method}: {opt_result.fun}')

    from espaloma.utils.samplers import langevin

    logprobfun = lambda x: - fun(x)
    gradlogprobfun = lambda x: - jac(x)
    x0 = opt_result.x
    v0 = onp.random.randn(*x0.shape)
    langevin_traj, langevin_log_prob_traj = langevin(x0, v0, logprobfun, gradlogprobfun, stepsize=1e-2, collision_rate=1e1, n_steps=10000)


    plt.plot(-langevin_log_prob_traj)
    plt.xlabel('langevin iteration')
    plt.ylabel('loss')
    plt.title(f'initialized from {method} result')
    plt.savefig(f'plots/{name}_{target_name}_langevin_loss_traj_alt_param.png', dpi=300)
    plt.close()


    # fictitious "temperature" -- from scipy.optimize.basinhopping documentation:
    #   The “temperature” parameter for the accept or reject criterion.
    #   Higher “temperatures” mean that larger jumps in function value will be accepted.
    #   For best results T should be comparable to the separation (in function value) between local minima.
    bh_temperature = 10.0

    opt_result = basinhopping(fun, params, T=bh_temperature, take_step=MultiplicativeNoiseStep(),
                              minimizer_kwargs=dict(method=method, jac=jac, callback=callback, options=min_options),
                              callback=bh_callback, disp=True)


    plot_residuals(predict(opt_result.x), target, mol_name=name, target_name=target_name)

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
    plt.savefig(f'plots/{name}_{target_name}_langevin_loss_traj_alt_param.png', dpi=300)
    plt.close()
