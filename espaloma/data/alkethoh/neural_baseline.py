# TODO: change my mind about where everything should go
# TODO: replace some hard-coded snapshot-batch logic with jax vmap

import mdtraj as md
import numpy as onp
from jax import numpy as np

data_path = ''
path_to_offmols = data_path + 'AlkEthOH_rings_offmols.pkl'
from pickle import load

with open(path_to_offmols, 'rb') as f:
    offmols = load(f)


def get_snapshots_and_energies(name='AlkEthOH_r1155'):
    snapshots_path = data_path + 'snapshots_and_energies/{}_molecule_traj.h5'.format(name)
    energies_path = data_path + 'snapshots_and_energies/{}_molecule_energies.npy'.format(name)
    ani1ccx_energies_path = data_path + 'snapshots_and_energies/{}_ani1ccx_energies.npy'.format(name)

    snapshots = md.load(snapshots_path)
    energies = onp.load(energies_path)
    ani1ccx_energies = onp.load(ani1ccx_energies_path)

    return snapshots, energies, ani1ccx_energies


## Loading atom and bond features:

# Atoms: one-hot encoding of element
atom_to_index = {'C': 0, 'H': 1, 'O': 2}


def atom_features_from_offmol(offmol):
    n_elements = len(atom_to_index)
    elements = [atom.element.symbol for atom in offmol.atoms]

    atoms = onp.zeros((len(elements), n_elements))
    for i in range(len(elements)):
        atoms[i][atom_to_index[elements[i]]] = 1
    return atoms


# Bonds: bond order and aromaticity
def bond_features_from_offmol(offmol):
    n_bonds = len(offmol.bonds)
    n_bond_features = 2
    features = onp.zeros((n_bonds, n_bond_features))

    for bond in offmol.bonds:
        i = bond.molecule_bond_index
        features[i, 0] = bond.bond_order
        features[i, 1] = bond.is_aromatic
    return features


def bond_index_dict_from_offmol(offmol):
    bond_index_dict = dict()

    for bond in offmol.bonds:
        a, b = bond.atom1_index, bond.atom2_index
        i = bond.molecule_bond_index
        bond_index_dict[(a, b)] = i
        bond_index_dict[(b, a)] = i
    return bond_index_dict


atom_dim = len(atom_to_index)  # one-hot
bond_dim = 2  # bond order, aromaticity

atom_param_dim = 4  # sigma, epsilon, electronegativity, hardness
bond_param_dim = 2  # k, r0
angle_param_dim = 2  # k, theta0

n_periodicities = 6
periodicities = np.arange(n_periodicities) + 1
torsion_param_dim = 2 * n_periodicities  # ks, phases


def initialize(layer_sizes, init_scale=0.1):
    return [(init_scale * onp.random.randn(m, n),  # weight matrix
             init_scale * onp.random.randn(n))  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


hidden_layer_dims = [64] * 3

get_layer_sizes = lambda input_dim, output_dim: [input_dim] + hidden_layer_dims + [output_dim]

f_1_layer_sizes = get_layer_sizes(atom_dim, atom_param_dim)
f_2_layer_sizes = get_layer_sizes(2 * atom_dim + bond_dim, bond_param_dim)
f_3_layer_sizes = get_layer_sizes(3 * atom_dim + 2 * bond_dim, angle_param_dim)
f_4_layer_sizes = get_layer_sizes(4 * atom_dim + 3 * bond_dim, torsion_param_dim)

f_1_params = initialize(f_1_layer_sizes)
f_2_params = initialize(f_2_layer_sizes)
f_3_params = initialize(f_3_layer_sizes)
f_4_params = initialize(f_4_layer_sizes)


def predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs


f_1 = lambda x: predict(f_1_params, x)
f_2 = lambda x: predict(f_2_params, x)
f_3 = lambda x: predict(f_3_params, x)
f_4 = lambda x: predict(f_4_params, x)

from jax import jit


@jit
def f_atom(f_1_params, x):
    """x = a1"""
    return predict(f_1_params, x)


@jit
def f_bond(f_2_params, x):
    """x = np.hstack([a1, b12, a2])"""
    return 0.5 * (predict(f_2_params, x) + predict(f_2_params, x[::-1]))


@jit
def f_angle(f_3_params, x):
    """x = np.hstack([a1, b12, a2, b23, a3])"""
    return 0.5 * (predict(f_3_params, x) + predict(f_3_params, x))


@jit
def f_torsion(f_4_params, x):
    """x = np.hstack([a1, b12, a2, b23, a3, b34, a4])"""
    return 0.5 * (predict(f_4_params, x) + predict(f_4_params, x))


from functools import lru_cache


@lru_cache(maxsize=len(offmols))
def extract_bond_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    b12 = bond_features_from_offmol(offmol)

    a1_inds = onp.array([bond.atom1_index for bond in offmol.bonds])
    a2_inds = onp.array([bond.atom2_index for bond in offmol.bonds])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]

    x = np.hstack((a1, b12, a2))
    inds = np.vstack((a1_inds, a2_inds)).T

    return x, inds


@lru_cache(maxsize=len(offmols))
def extract_angle_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    bond_features = bond_features_from_offmol(offmol)
    bond_index_dict = bond_index_dict_from_offmol(offmol)

    angles = list(offmol.angles)  # offmol.angles is a set of 3-tuples of atoms

    a1_inds = onp.array([a.molecule_atom_index for (a, _, _) in angles])
    a2_inds = onp.array([b.molecule_atom_index for (_, b, _) in angles])
    a3_inds = onp.array([c.molecule_atom_index for (_, _, c) in angles])

    b12_inds = onp.array([bond_index_dict[(a.molecule_atom_index, b.molecule_atom_index)] for (a, b, _) in angles])
    b23_inds = onp.array([bond_index_dict[(b.molecule_atom_index, c.molecule_atom_index)] for (_, b, c) in angles])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]
    a3 = atoms[a3_inds]

    b12 = bond_features[b12_inds]
    b23 = bond_features[b23_inds]

    x = np.hstack((a1, b12, a2, b23, a3))
    inds = np.vstack((a1_inds, a2_inds, a3_inds)).T

    return x, inds


@lru_cache(maxsize=len(offmols))
def extract_torsion_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    bond_features = bond_features_from_offmol(offmol)
    bond_index_dict = bond_index_dict_from_offmol(offmol)

    torsions = list(offmol.propers)  # offmol.propers is a set of 4-tuples of atoms

    a1_inds = onp.array([a.molecule_atom_index for (a, _, _, _) in torsions])
    a2_inds = onp.array([b.molecule_atom_index for (_, b, _, _) in torsions])
    a3_inds = onp.array([c.molecule_atom_index for (_, _, c, _) in torsions])
    a4_inds = onp.array([d.molecule_atom_index for (_, _, _, d) in torsions])

    b12_inds = onp.array([bond_index_dict[(a.molecule_atom_index, b.molecule_atom_index)] for (a, b, _, _) in torsions])
    b23_inds = onp.array([bond_index_dict[(b.molecule_atom_index, c.molecule_atom_index)] for (_, b, c, _) in torsions])
    b34_inds = onp.array([bond_index_dict[(c.molecule_atom_index, d.molecule_atom_index)] for (_, _, c, d) in torsions])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]
    a3 = atoms[a3_inds]
    a4 = atoms[a3_inds]

    b12 = bond_features[b12_inds]
    b23 = bond_features[b23_inds]
    b34 = bond_features[b34_inds]

    x = np.hstack((a1, b12, a2, b23, a3, b34, a4))
    inds = np.vstack((a1_inds, a2_inds, a3_inds, a4_inds)).T

    return x, inds


## Compute energies

# Bonds
@jit
def compute_distances(xyz, pair_inds):
    """
    xyz.shape : (n_snapshots, n_atoms, n_dim)
    pair_inds.shape : (n_pairs, 2)
    """

    diffs = (xyz[:, pair_inds[:, 0]] - xyz[:, pair_inds[:, 1]])
    return np.sqrt(np.sum(diffs ** 2, axis=2))


@jit
def harmonic_bond_potential(r, k, r0):
    return 0.5 * k * (r0 - r) ** 2


def compute_harmonic_bond_potential(offmol, xyz, f_2_params):
    x, pair_inds = extract_bond_term_inputs(offmol)
    r = compute_distances(xyz, pair_inds)
    k, r0 = f_bond(f_2_params, x).T
    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


# Angles
@jit
def angle(a, b, c):
    """a,b,c each have shape (n_snapshots, n_angles, dim)"""

    u = b - a
    u /= np.sqrt(np.sum(u ** 2, axis=2))[:, :, np.newaxis]

    v = c - b
    v /= np.sqrt(np.sum(v ** 2, axis=2))[:, :, np.newaxis]

    udotv = np.sum(u * v, axis=2)

    return np.arccos(-udotv)


@jit
def compute_angles(xyz, inds):
    a, b, c = xyz[:, inds[:, 0]], xyz[:, inds[:, 1]], xyz[:, inds[:, 2]]
    return angle(a, b, c)


def harmonic_angle_potential(theta, k, theta0):
    return 0.5 * k * (theta0 - theta) ** 2


def compute_harmonic_angle_potential(offmol, xyz, f_3_params):
    x, inds = extract_angle_term_inputs(offmol)
    theta = compute_angles(xyz, inds)
    k, theta0 = f_angle(f_3_params, x).T
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)


# Torsions
@jit
def dihedral_angle(a, b, c, d):
    b1 = b - a
    # b2 = c - b # mdtraj convention
    b2 = b - c  # openmm convention
    b3 = d - c

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = np.sum(b1 * c1, axis=2) * np.sum(b2 * b2, axis=2) ** 0.5
    p2 = np.sum(c1 * c2, axis=2)

    return np.arctan2(p1, p2)


@jit
def compute_torsions(xyz, inds):
    a, b, c, d = xyz[:, inds[:, 0]], xyz[:, inds[:, 1]], xyz[:, inds[:, 2]], xyz[:, inds[:, 3]]
    return dihedral_angle(a, b, c, d)


@jit
def periodic_torsion_potential(theta, ks, phases, periodicities):
    return np.sum([ks[:, i] * (1 + np.cos(periodicities[:, i] * theta - phases[:, i])) for i in range(n_periodicities)],
                  axis=0)


def compute_periodic_torsion_potential(offmol, xyz, f_4_params):
    x, inds = extract_torsion_term_inputs(offmol)
    theta = compute_torsions(xyz, inds)

    params = f_torsion(f_4_params, x)
    ks, phases = params[:, :n_periodicities], params[:, n_periodicities:]

    # TODO; clean this up a bit
    periodicities_ = np.array([periodicities for _ in ks])

    return np.sum(periodic_torsion_potential(theta, ks, phases, periodicities_), axis=1)


# Nonbonded
from jax import vmap


def pdist(x):
    """should be consistent with scipy.spatial.pdist:
    flat, non-redundant pairwise distances"""
    diffs = np.expand_dims(x, 1) - np.expand_dims(x, 0)
    squared_distances = np.sum(diffs ** 2, axis=2)
    inds = onp.triu_indices_from(squared_distances, k=1)
    return np.sqrt(squared_distances[inds])


def pred_nonbonded_energy(offmol, xyz, f_1_params):
    x = atom_features_from_offmol(offmol)

    distances = vmap(pdist)(xyz)
    atom_params = f_atom(f_1_params, x)

    # TODO: finish this thought

    raise (NotImplementedError)


# Altogether
def pred_valence_energy(offmol, xyz, f_2_params, f_3_params, f_4_params):
    bond_energy = compute_harmonic_bond_potential(offmol, xyz, f_2_params)
    angle_energy = compute_harmonic_angle_potential(offmol, xyz, f_3_params)
    torsion_energy = compute_periodic_torsion_potential(offmol, xyz, f_4_params)

    return bond_energy + angle_energy + torsion_energy


# openmm
from simtk import unit
from simtk import openmm as mm
from simtk.openmm.app import Simulation
from simtk.openmm import XmlSerializer


def get_sim(name):
    """nonbonded forces in group 0, all other forces in group 1"""
    mol = offmols[name]

    # Parametrize the topology and create an OpenMM System.
    topology = mol.to_topology()

    with open(data_path + 'snapshots_and_energies/{}_system.xml'.format(name), 'r') as f:
        xml = f.read()

    system = XmlSerializer.deserializeSystem(xml)

    platform = mm.Platform.getPlatformByName('Reference')
    integrator = mm.VerletIntegrator(1.0)

    sim = Simulation(topology, system, integrator, platform=platform)

    inds_of_nb_forces = [i for i in range(sim.system.getNumForces()) if
                         'Nonbonded' in sim.system.getForce(i).__class__.__name__]

    for i in range(sim.system.getNumForces()):
        sim.system.getForce(i).setForceGroup(1)
    for i in inds_of_nb_forces:
        sim.system.getForce(i).setForceGroup(0)

    return sim


def set_positions(sim, pos):
    sim.context.setPositions(pos)


def get_energy(sim):
    return sim.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole


def get_nb_energy(sim):
    """assumes NonbondedForce is in group 0"""
    return sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy() / unit.kilojoule_per_mole


if __name__ == '__main__':
    name = 'AlkEthOH_r33' # easy example
    # name = 'AlkEthOH_r1155' # difficult example
    offmol = offmols[name]
    snapshots, _, _ = get_snapshots_and_energies(name)
    xyz = snapshots.xyz

    all_params = f_2_params, f_3_params, f_4_params

    valence_energies = []
    sim = get_sim(name)
    for conf in xyz:
        set_positions(sim, conf * unit.nanometer)
        U_tot = get_energy(sim)
        U_nb = get_nb_energy(sim)
        valence_energies.append(U_tot - U_nb)
    valence_target = np.array(valence_energies)


    def loss(all_params):
        f_2_params, f_3_params, f_4_params = all_params
        U_valence = pred_valence_energy(offmol, xyz, f_2_params, f_3_params, f_4_params)
        return np.sum((valence_target - U_valence) ** 2)


    from jax import grad

    from time import time

    t0 = time()
    g = grad(loss)(all_params)
    t1 = time()
    g = grad(loss)(all_params)
    t2 = time()

    print(f'time to compile gradient: {t1 - t0:.3f}s')
    print(f'time to compute gradient: {t2 - t1:.3f}s')

    # TODO: this, but less ugly-ly
    #   didn't see where Jax has equivalent of autograd.misc.flatten)
    #   naively mixing autograd and jax code leads to type-errors within autograd:
    #       TypeError: Can't differentiate w.r.t. type <class 'jax.interpreters.xla.DeviceArray'>
    from autograd.misc import flatten


    def net_to_numpy(net_params):
        return [(onp.array(W), onp.array(b)) for (W, b) in net_params]


    def all_to_numpy(all_params):
        return [net_to_numpy(net_params) for net_params in all_params]


    flat_params, unflatten = flatten(all_params)


    # Defining functions that can talk to scipy optimizers...
    def fun(flat_params):
        all_params = unflatten(flat_params)
        return float(loss(all_params))


    def jac(flat_params):
        all_params = unflatten(flat_params)
        g = grad(loss)(all_params)
        g_flat, _ = flatten(all_to_numpy(g))
        return onp.array(g_flat, dtype=onp.float64)


    from scipy.optimize import minimize

    opt_result = minimize(fun, x0=flat_params, jac=jac, method='L-BFGS-B', options=dict(disp=True))
    print(opt_result)
