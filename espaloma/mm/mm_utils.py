import numpy as onp
from jax import jit
from jax import numpy as np
from simtk import unit

from espaloma.utils.symmetry import get_unique_torsions, get_unique_bonds, canonicalize_order, get_unique_angles

energy_unit = unit.kilojoule_per_mole
force_unit = unit.kilojoule_per_mole / unit.nanometer


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
    n_periodicities = phases.shape[1]
    return np.sum([ks[:, i] * (1 + np.cos(periodicities[:, i] * theta - phases[:, i])) for i in range(n_periodicities)],
                  axis=0)


n_periodicities = 6
periodicities = np.arange(n_periodicities) + 1


def compute_periodic_torsion_potential(xyz, params, quad_inds, torsion_inds):
    """

    :param xyz:
    :param params:
        length ( 2 * n_unique * n_periodicities )
    :param quad_inds:
    :param torsion_inds:
    :return:
    """
    theta = compute_torsions(xyz, quad_inds)

    n_unique = int(len(params) / (2 * n_periodicities))
    params = np.reshape(params, (n_unique, (2 * n_periodicities)))

    ks, phases = params[torsion_inds][:, :n_periodicities], params[torsion_inds][:, n_periodicities:]

    # TODO; clean this up a bit
    periodicities_ = np.array([periodicities for _ in ks])

    return np.sum(periodic_torsion_potential(theta, ks, phases, periodicities_), axis=1)




# Can represent springs a couple different ways
# 1. In terms of spring constant and equilibrium length
#   u(r; k, r0) = k/2 (r - r0)^2
# 2. In terms of a fixed interval of allowable equilibrium lengths [r1, r2]
#   u(r; k1, k2) = k1 (r - r1)^2 + k2 (r - r2)^2
#   This has the benefit of allowing us to express variable equilibrium length in terms of
#   fixed design matrix [(r - r1)^2, (r - r2)^2] and variable coefficients [k1, k2].
from jax import jit


@jit
def harmonic_bond_potential_alt_param(r, k1, k2, r1, r2):
    return k1 * (r - r1) ** 2 + k2 * (r - r2) ** 2





@jit
def harmonic_angle_potential_alt_param(theta, k1, k2, theta1, theta2):
    return k1 * (theta - theta1) ** 2 + k2 * (theta - theta2) ** 2


def compute_harmonic_bond_potential(xyz, params, pair_inds, bond_inds):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param pair_inds:
        array of shape (len(offmol.bonds), 2)
    :param bond_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    ks, r0s = params[:n_unique], params[n_unique:]
    k, r0 = ks[bond_inds], r0s[bond_inds]

    r = compute_distances(xyz, pair_inds)
    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


def compute_harmonic_bond_potential_alt_param(xyz, params, pair_inds, bond_inds, r1=0.0, r2=0.3):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param pair_inds:
        array of shape (len(offmol.bonds), 2)
    :param bond_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :param r1, float
        min representable bond length
    :param r2, float
        max representable bond length
    :return:
    """

    n_unique = int(len(params) / 2)
    k1s, k2s = params[:n_unique], params[n_unique:]
    k1, k2 = k1s[bond_inds], k2s[bond_inds]

    k = k1 + k2
    r0 = ((k1 * r1) + (k2 * r2)) / (k1 + k2)

    r = compute_distances(xyz, pair_inds)


    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


def compute_harmonic_angle_potential(xyz, params, triple_inds, angle_inds):
    """

    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param triple_inds:
        array of shape (len(offmol.angles), 3)
    :param angle_inds:
        numpy array of length len(offmol.angles),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    ks, theta0s = params[:n_unique], params[n_unique:]
    # Jax: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    #angle_inds = tuple(angle_inds)
    angle_inds = np.array(angle_inds)
    #print(ks)
    #print(angle_inds)

    k = ks[angle_inds]
    theta0 = theta0s[angle_inds]
    # theta0s: size 23
    # params: size 46
    # ks: size 23

    theta = compute_angles(xyz, triple_inds)
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)





def compute_harmonic_angle_potential_alt_param(xyz, params, triple_inds, angle_inds, theta1=0, theta2=np.pi):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param triple_inds:
        array of shape (len(offmol.bonds), 2)
    :param angle_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :param theta1
        min representable angle
    :param theta2
        max representable angle
    :return:
    """

    n_unique = int(len(params) / 2)
    k1s, k2s = params[:n_unique], params[n_unique:]
    k1, k2 = k1s[angle_inds], k2s[angle_inds]

    k = k1 + k2
    theta0 = ((k1 * theta1) + (k2 * theta2)) / (k1 + k2)

    theta = compute_angles(xyz, triple_inds)
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)


# Nonbonded


def pdist(x):
    """should be consistent with scipy.spatial.pdist:
    flat, non-redundant pairwise distances"""
    diffs = np.expand_dims(x, 1) - np.expand_dims(x, 0)
    squared_distances = np.sum(diffs ** 2, axis=2)
    inds = onp.triu_indices_from(squared_distances, k=1)
    return np.sqrt(squared_distances[inds])


# openmm
from simtk import unit
from simtk import openmm as mm
from simtk.openmm.app import Simulation
from simtk.openmm import XmlSerializer
from pkg_resources import resource_filename
from espaloma.data.alkethoh.data import offmols


def get_sim(name):
    """
    harmonicbondforce in group 0
    harmonicangleforce in group 1
    periodictorsionforce in group 2
    nonbondedforce in group 3
    anything else in group 4
    """
    mol = offmols[name]

    # Parametrize the topology and create an OpenMM System.
    topology = mol.to_topology()

    fname = resource_filename('espaloma.data.alkethoh', 'snapshots_and_energies/{}_system.xml'.format(name))

    with open(fname, 'r') as f:
        xml = f.read()

    system = XmlSerializer.deserializeSystem(xml)

    platform = mm.Platform.getPlatformByName('Reference')
    integrator = mm.VerletIntegrator(1.0)

    sim = Simulation(topology, system, integrator, platform=platform)

    for i in range(sim.system.getNumForces()):
        class_name = sim.system.getForce(i).__class__.__name__
        if 'HarmonicBond' in class_name:
            sim.system.getForce(i).setForceGroup(0)
        elif 'HarmonicAngle' in class_name:
            sim.system.getForce(i).setForceGroup(1)
        elif 'PeriodicTorsion' in class_name:
            sim.system.getForce(i).setForceGroup(2)
        elif 'Nonbonded' in class_name:
            sim.system.getForce(i).setForceGroup(3)
        else:
            print('un-recognized force, assigned to group 4')
            sim.system.getForce(i).setForceGroup(4)

    return sim


def set_positions(sim, pos):
    sim.context.setPositions(pos)


def get_energy(sim):
    return sim.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole


def get_energy_by_group(sim, group=0):
    return sim.context.getState(getEnergy=True, groups={group}).getPotentialEnergy() / energy_unit


def get_force_by_group(sim, group=0):
    return sim.context.getState(getForces=True, groups={group}).getForces(asNumpy=True) / force_unit


def get_force(sim):
    return sim.context.getState(getForces=True).getForces(asNumpy=True) / force_unit


def get_bond_energy(sim):
    """assumes HarmonicBondForce is in group 0"""
    return get_energy_by_group(sim, 0)


def get_bond_force(sim):
    """assumes HarmonicBondForce is in group 0"""
    return get_force_by_group(sim, 0)


def get_angle_energy(sim):
    """assumes HarmonicAngleForce is in group 1"""
    return get_energy_by_group(sim, 1)


def get_angle_force(sim):
    """assumes HarmonicAngleForce is in group 1"""
    return get_force_by_group(sim, 1)


def get_torsion_energy(sim):
    """assumes PeriodicTorsionForce is in group 2"""
    return get_energy_by_group(sim, 2)


def get_torsion_force(sim):
    """assumes PeriodicTorsionForce is in group 2"""
    return get_force_by_group(sim, 2)


# nb vs. valence

def get_nb_energy(sim):
    """assumes NonbondedForce is in group 3"""
    return get_energy_by_group(sim, 3)


def get_nb_force(sim):
    """assumes NonbondedForce is in group 3"""
    return get_force_by_group(sim, 3)

def get_valence_energy(sim):
    return get_energy(sim) - get_nb_energy(sim)


def get_valence_force(sim):
    return get_force(sim) - get_nb_force(sim)



from collections import namedtuple

MMComponents = namedtuple('MMComponents',
                          field_names=['bonds', 'angles', 'torsions', 'valence', 'nonbonded', 'total'],
                          defaults=[False, False, False, False, False, False])
default_components = MMComponents()

get_mm_energies = dict(bonds=get_bond_energy, angles=get_angle_energy, torsions=get_torsion_energy,
                       valence=get_valence_energy, nonbonded=get_nb_energy, total=get_energy)
get_mm_forces = dict(bonds=get_bond_force, angles=get_angle_force, torsions=get_torsion_force,
                     valence=get_valence_force, nonbonded=get_nb_force, total=get_force)


from espaloma.data.alkethoh.ani import  get_snapshots_energies_and_forces

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



def initialize_bonds(sim, offmol, noise_magnitude=1.0):
    # bonds
    pair_inds, bond_inds = get_unique_bonds(offmol)
    n_unique_bonds = len(set(bond_inds))
    n_bond_params = 2 * n_unique_bonds
    bond_params = onp.zeros(n_bond_params)

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
        multiplicative_noise = 2 * noise_magnitude * (
                    onp.random.rand(2) - 0.5) + 1.0  # uniform between [1-noise_magnitude, 1+noise_magnitude]
        bond_params[bond_inds[i]] = k_ * multiplicative_noise[0]
        bond_params[bond_inds[i] + n_unique_bonds] = length_ * multiplicative_noise[1]

    return bond_params


def initialize_angles(sim, offmol, noise_magnitude=1.0):
    triple_inds, angle_inds = get_unique_angles(offmol)
    n_unique_angles = len(set(angle_inds))
    n_angle_params = 2 * n_unique_angles
    angle_params = onp.zeros(n_angle_params)

    harmonic_angle_force = [f for f in sim.system.getForces() if ("HarmonicAngle" in f.__class__.__name__)][0]
    omm_angle_params = dict()

    for i in range(harmonic_angle_force.getNumAngles()):
        a, b, c, theta, k = harmonic_angle_force.getAngleParameters(i)
        tup = canonicalize_order((a, b, c))
        omm_angle_params[tup] = (theta, k)

    for i in range(len(triple_inds)):
        theta, k = omm_angle_params[tuple(triple_inds[i])]
        theta_, k_ = theta / unit.radian, k / (unit.kilojoule_per_mole / (unit.radian ** 2))

        multiplicative_noise = 2 * noise_magnitude * (
                onp.random.rand(2) - 0.5) + 1.0  # uniform between [1-noise_magnitude, 1+noise_magnitude]

        angle_params[angle_inds[i]] = k_ * multiplicative_noise[0]
        angle_params[angle_inds[i] + n_unique_angles] = theta_ * multiplicative_noise[1]

    return angle_params


def initialize_torsions(sim, offmol, noise_magnitude=1.0):

    quad_inds, torsion_inds = get_unique_torsions(offmol)
    n_unique_torsions = len(set(torsion_inds))
    n_torsion_params = 2 * n_unique_torsions

    periodic_torsion_force = [f for f in sim.system.getForces() if ("PeriodicTorsion" in f.__class__.__name__)][0]
    omm_torsion_params = dict()
    for i in range(periodic_torsion_force.getNumTorsions()):
        # TODO: actually initialize at or near underlying values, rather than initializing at 0
        a, b, c, d, periodicity, phase, k = periodic_torsion_force.getTorsionParameters(i)
        k_ = k / (unit.kilojoule_per_mole / (unit.radian**2))

    _, torsion_inds = get_unique_torsions(offmol)
    n_unique_torsions = len(set(torsion_inds))
    n_torsion_params = 2 * n_unique_torsions * n_periodicities
    torsion_params = onp.zeros(n_torsion_params)

    return torsion_params
