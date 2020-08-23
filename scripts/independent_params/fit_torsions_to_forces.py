# separate parameters for every atom, torsion, torsion, torsion, up to symmetry

from time import time

import matplotlib.pyplot as plt
import numpy as onp
from jax.config import config  # TODO: is there a way to enable this globally?

from scripts.independent_params.plots import plot_residuals

config.update("jax_enable_x64", True)

from espaloma.utils.jax import jax_play_nice_with_scipy
from jax import grad, jit, numpy as np
from scipy.optimize import basinhopping
from simtk import unit

from espaloma.data.alkethoh.ani import get_snapshots_energies_and_forces
from espaloma.data.alkethoh.data import offmols
from espaloma.mm.mm_utils import get_force_targets, MMComponents
from espaloma.mm.mm_utils import get_sim, compute_periodic_torsion_potential, n_periodicities
from espaloma.utils.symmetry import get_unique_torsions, canonicalize_order
onp.random.seed(1234)

# TODO: add coupling terms

# TODO: initializer classes
#   initialize at mean values vs. at openff values

def initialize_torsions(offmol, noise_magnitude=1.0):
    
    quad_inds, torsion_inds = get_unique_torsions(offmol)
    n_unique_torsions = len(set(torsion_inds))
    n_torsion_params = 2 * n_unique_torsions
    sim = get_sim(name)
    
    periodic_torsion_force = [f for f in sim.system.getForces() if ("PeriodicTorsion" in f.__class__.__name__)][0]
    omm_torsion_params = dict()
    for i in range(periodic_torsion_force.getNumTorsions()):
        a, b, c, d, periodicity, phase, k = periodic_torsion_force.getTorsionParameters(i)
        k_ = k / (unit.kilojoule_per_mole / (unit.radian**2))
    
    _, torsion_inds = get_unique_torsions(offmol)
    n_unique_torsions = len(set(torsion_inds))
    n_torsion_params = 2 * n_unique_torsions * n_periodicities
    torsion_params = onp.zeros(n_torsion_params)

    def unpack(params):
        return ([], [], params)

    return torsion_params, unpack


if __name__ == '__main__':
    # look at a single molecule first
    name = 'AlkEthOH_r4'
    offmol = offmols[name]

    params, unpack = initialize_torsions(offmol)
    quad_inds, torsion_inds = get_unique_torsions(offmol)

    # targets
    mm_components, ani1ccx_forces = get_force_targets(name, MMComponents(torsions=True))
    target = mm_components.torsions

    # trajectory
    traj, _, _ = get_snapshots_energies_and_forces(name)
    xyz = traj.xyz


    @jit
    def compute_torsion_force(xyz, torsion_params, quad_inds, torsion_inds):
        total_U = lambda xyz: np.sum(compute_periodic_torsion_potential(xyz, torsion_params, quad_inds, torsion_inds))
        return - grad(total_U)(xyz)


    @jit
    def predict(all_params):
        torsion_params, torsion_params, torsion_params = unpack(all_params)
        F_torsion = compute_torsion_force(xyz, torsion_params, quad_inds, torsion_inds)
        return F_torsion


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

    plot_residuals(predict(opt_result.x), target, name, 'torsion')

    loss_traj = [fun(theta) for theta in traj]
    running_min_loss_traj = onp.minimum.accumulate(loss_traj)
    plt.plot(running_min_loss_traj)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'{method} iteration (within basin-hopping)')
    plt.ylabel('running minimum of RMSE loss\n(predicted MM torsion force vs. OFF1.0 torsion force, in kJ/mol / nm)')
    plt.title(f'{name}: torsion force regression')
    plt.savefig(f'plots/{name}_torsion_loss_traj.png', bbox_inches='tight', dpi=300)
    plt.close()