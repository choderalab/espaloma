# separate parameters for every atom, bond, angle, torsion, up to symmetry

from time import time

import matplotlib.pyplot as plt
import numpy as onp
from jax.config import config  # TODO: is there a way to enable this globally?

config.update("jax_enable_x64", True)
from jax import grad, jit, numpy as np
from scipy.optimize import basinhopping
from simtk import unit

from espaloma.data.alkethoh.ani import get_snapshots_energies_and_forces
from espaloma.data.alkethoh.data import offmols
from espaloma.mm.mm_utils import get_energy, get_bond_energy, get_angle_energy, get_torsion_energy, \
    get_valence_energy, \
    get_nb_energy
from espaloma.mm.mm_utils import get_force, get_bond_force, get_angle_force, get_torsion_force, \
    get_valence_force, get_nb_force
from espaloma.mm.mm_utils import get_sim, set_positions, compute_harmonic_bond_potential
from espaloma.utils.symmetry import get_unique_bonds, canonicalize_order
onp.random.seed(1234)

# TODO: add coupling terms
# TODO: toggle between `compute_harmonic_bond_potential` and alternate parameterization

# TODO: initializer classes
#   initialize at mean values vs. at openff values

def initialize_bonds(offmol, noise_magnitude=0.5):
    # bonds
    pair_inds, bond_inds = get_unique_bonds(offmol)
    n_unique_bonds = len(set(bond_inds))
    n_bond_params = 2 * n_unique_bonds
    bond_params = onp.zeros(n_bond_params)

    sim = get_sim(name)

    # compare pair inds from harmonic_bond_force and autodiff'd one
    harmonic_bond_force = [f for f in sim.system.getForces() if ("HarmonicBond" in f.__class__.__name__)][0]
    omm_pair_inds = []
    omm_bond_params = dict()

    for i in range(harmonic_bond_force.getNumBonds()):
        a, b, length, k = harmonic_bond_force.getBondParameters(i)
        tup = canonicalize_order((a, b))
        omm_bond_params[tup] = (length, k)
        omm_pair_inds.append(tup)

    # assert that I'm defining bonds on the same pairs of atoms
    assert ((set(omm_pair_inds) == set([tuple(p) for p in pair_inds])))

    # What if I initialize with MM parameters
    for i in range(len(pair_inds)):
        length, k = omm_bond_params[tuple(pair_inds[i])]
        length_, k_ = length / unit.nanometer, k / (unit.kilojoule_per_mole / (unit.nanometer ** 2))
        multiplicative_noise = noise_magnitude * (
                    onp.random.rand(2) - 0.5) + 1.0  # uniform between [1-noise_magnitude, 1+noise_magnitude]
        bond_params[bond_inds[i]] = k_ * multiplicative_noise[0]
        bond_params[bond_inds[i] + n_unique_bonds] = length_ * multiplicative_noise[1]

    def unpack(params):
        return (params, [], [])

    return bond_params, unpack


# which energy/force components do I want to look at?
from collections import namedtuple

MMComponents = namedtuple('MMComponents',
                          field_names=['bonds', 'angles', 'torsions', 'valence', 'nonbonded', 'total'],
                          defaults=[False, False, False, False, False, False])
default_components = MMComponents()

get_mm_energies = dict(bonds=get_bond_energy, angles=get_angle_energy, torsions=get_torsion_energy,
                       valence=get_valence_energy, nonbonded=get_nb_energy, total=get_energy)
get_mm_forces = dict(bonds=get_bond_force, angles=get_angle_force, torsions=get_torsion_force,
                     valence=get_valence_force, nonbonded=get_nb_force, total=get_force)


def get_energy_targets(name, components=default_components):
    """

    :param name:
    :param components:
    :return:
    """
    traj, ani1ccx_energies, _ = get_snapshots_energies_and_forces(name)
    xyz = traj.xyz
    sim = get_sim(name)

    # Define energy targets
    component_names = [component for component in components._fields if components.__getattribute__(component)]

    # initialize energies_dict
    energies = dict(zip(component_names, [list() for _ in component_names]))
    for component in components._fields:
        if not components.__getattribute__(component):
            energies[component] = None

    # fill in energies
    for conf in xyz:
        set_positions(sim, conf * unit.nanometer)
        for component in component_names:
            energies[component].append(get_mm_energies[component](sim))
    # convert from list of floats to np array
    for component in component_names:
        energies[component] = np.array(energies[component])

    # return named tuple
    mm_components = MMComponents(**energies)
    return mm_components, ani1ccx_energies


def get_force_targets(name, components=default_components):
    traj, _, ani1ccx_forces = get_snapshots_energies_and_forces(name)
    xyz = traj.xyz
    sim = get_sim(name)

    # Define force targets
    component_names = [component for component in components._fields if components.__getattribute__(component)]

    # initialize forces_dict
    forces = dict(zip(component_names, [list() for _ in component_names]))
    for component in components._fields:
        if not components.__getattribute__(component):
            forces[component] = None

    # fill in forces
    for conf in xyz:
        set_positions(sim, conf * unit.nanometer)
        for component in component_names:
            forces[component].append(get_mm_forces[component](sim))
    # convert from list of arrays to single array
    for component in component_names:
        forces[component] = np.array(forces[component])

    # return named tuple
    mm_components = MMComponents(**forces)
    return mm_components, ani1ccx_forces


def plot_residuals(predicted, target, mol_name, target_name):
    scatter_kwargs = dict(s=1, alpha=0.5)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(predicted, target, **scatter_kwargs)
    unit = '(kJ/mol / nm^2)'
    plt.xlabel(f'predicted {unit}')
    plt.ylabel(f'target {unit}')
    plt.title(f'{mol_name}: {target_name} force')

    plt.subplot(1, 2, 2)
    plt.scatter(predicted, predicted - target, **scatter_kwargs)
    plt.xlabel(f'predicted {unit}')
    plt.ylabel(f'predicted - target {unit}')
    plt.title(f'{mol_name}: {target_name} force residuals')
    plt.tight_layout()

    plt.savefig(f'plots/{mol_name}_{target_name}_residuals.png', bbox_inches='tight', dpi=300)
    plt.close()


def jax_play_nice_with_scipy(jax_loss_fxn):
    """Defining functions that can talk to scipy optimizers"""

    def fun(params):
        return float(jax_loss_fxn(params))  # make sure return type is a plain old float)

    g = grad(jax_loss_fxn)

    def jac(params):
        return onp.array(g(params), dtype=onp.float64)  # make sure return type is a float64 array

    return fun, jac


if __name__ == '__main__':
    # look at a single molecule first
    name = 'AlkEthOH_r4'
    offmol = offmols[name]
    # params, unpack = initialize_off_params(offmol)
    params, unpack = initialize_bonds(offmol)
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
        bond_params, angle_params, torsion_params = unpack(all_params)
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
    g = grad(loss);
    t0 = time();
    _ = g(params);
    t1 = time();
    _ = g(params);
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

    plot_residuals(predict(opt_result.x), target, name, 'bond')

    loss_traj = [fun(theta) for theta in traj]
    running_min_loss_traj = onp.minimum.accumulate(loss_traj)
    plt.plot(running_min_loss_traj)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'{method} iteration (within basin-hopping)')
    plt.ylabel('running minimum of RMSE loss\n(predicted MM bond force vs. OFF1.0 bond force, in kJ/mol / nm^2)')
    plt.title('{name}: bond force regression')
    plt.savefig(f'plots/{name}_bond_loss_traj.png', bbox_inches='tight', dpi=300)
    plt.close()
