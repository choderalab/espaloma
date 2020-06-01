# TODO: one molecule dataset -- configurations only

# TODO: AlkEthOH dataset -- well-sampled

# TODO: parm@frosst set -- bigger set

from time import time

import numpy as np
from openforcefield.typing.engines.smirnoff import ForceField
from openmmtools.integrators import BAOABIntegrator
from simtk import openmm as mm
from simtk import unit
from simtk.openmm.app import Simulation
from tqdm import tqdm

distance_unit = unit.nanometer
energy_unit = unit.kilojoule_per_mole
force_unit = energy_unit / (distance_unit ** 2)

temperature = 500 * unit.kelvin
stepsize = 1 * unit.femtosecond
collision_rate = 1 / unit.picosecond

# Load the OpenFF "Parsley" force field. -- pick unconstrained so that Hbond stretches are sampled...
forcefield = ForceField('openff_unconstrained-1.0.0.offxml')


def sim_from_mol(mol):
    # Parametrize the topology and create an OpenMM System.
    topology = mol.to_topology()
    system = forcefield.create_openmm_system(topology)

    platform = mm.Platform.getPlatformByName('Reference')
    integrator = BAOABIntegrator(
        temperature=temperature,
        collision_rate=collision_rate,
        timestep=stepsize)

    sim = Simulation(topology, system, integrator, platform=platform)

    mol.generate_conformers()
    sim.context.setPositions(mol.conformers[0])
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature)
    return sim


def set_positions(sim, pos):
    sim.context.setPositions(pos)


def get_energy(sim):
    return sim.context.getState(getEnergy=True).getPotentialEnergy()


def get_positions(sim):
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_force(sim):
    return sim.context.getState(getForce=True).getForce()


def collect_samples(sim, n_samples=1000, n_steps_per_sample=100):
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim).value_in_unit(distance_unit))
    return np.array(samples)


def get_energies(sim, xyz):
    energy_list = []

    for pos in xyz:
        set_positions(sim, pos)

        energy_list.append(get_energy(sim).value_in_unit(energy_unit))
    return np.array(energy_list)


def get_forces(sim, xyz):
    force_list = []

    for pos in xyz:
        set_positions(sim, pos)

        force_list.append(get_force(sim).value_in_unit(force_unit))
    return np.array(force_list)


def create_entry(name, mol, n_samples, n_steps_per_sample):
    prefix = 'snapshots_and_energies/{}_'.format(name)

    print('creating openmm sim')
    t0 = time()
    sim = sim_from_mol(mol)
    t1 = time()
    print('that took {:.3}s'.format(t1 - t0))

    print('collecting samples')
    t0 = time()
    xyz_in_nm = collect_samples(sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample)
    t1 = time()
    print('that took {:.3}s'.format(t1 - t0))

    print('saving traj')
    import mdtraj as md

    traj = md.Trajectory(xyz_in_nm, mol.to_topology().to_openmm())
    traj.save_hdf5(prefix + 'molecule_traj.h5')

    # energy arrays and regression target, in kJ/mol
    mm_energy_array = get_energies(sim, xyz_in_nm)

    # TODO: also forces
    np.save(prefix + '_molecule_energies.npy', mm_energy_array)


if __name__ == '__main__':

    # n_samples, n_steps_per_sample = 1000, 1000
    n_samples, n_steps_per_sample = 100, 100

    from pickle import load

    with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
        mols = load(f)

    names = list(mols.keys())
    successes, failures = [], []
    for name in tqdm(names):
        try:
            create_entry(name, mols[name], n_samples, n_steps_per_sample)
            successes.append(name)
        except Exception as e:
            print('something failed for some reason!')
            print(e)
            failures.append(name)

    print('# sucesses: {}'.format(len(successes)))
    print('# failures: {}'.format(len(failures)))

    print('successes: ', successes)
    print('failures: ', failures)
