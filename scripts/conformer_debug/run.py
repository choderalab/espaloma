import numpy as np
from simtk import openmm, unit
from openmmforcefields.generators import SystemGenerator
from simtk.openmm.app import Simulation
from simtk.unit import Quantity
from openforcefield.topology import Molecule

TEMPERATURE = 300 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond

def run():
    mol = Molecule.from_smiles(
        "[H][O][S]1(=[O])=[N][C]2=[C]([C]([N]([H])[H])=[N]1)[N]([H])[C]([c]1[c]([H])[c]([H])[c]([H])[c]([H])[c]1[H])=[N]2",
    )

    # assign partial charge
    # mol.assign_partial_charges("am1bcc")

    # parameterize topology
    topology = mol.to_topology().to_openmm()

    generator = SystemGenerator(
        small_molecule_forcefield="gaff-1.81",
        molecules=[mol],
    )

    # create openmm system
    system = generator.create_system(
        topology,
    )

    # use langevin integrator
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE,
    )

    # initialize simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator,
        platform=openmm.Platform.getPlatformByName("Reference"),
    )

    import openforcefield

    # get conformer
    mol.generate_conformers(
        toolkit_registry=openforcefield.utils.RDKitToolkitWrapper(),
    )

    # put conformer in simulation
    simulation.context.setPositions(mol.conformers[0])

    # set velocities
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)

    # initialize empty list for samples.
    samples = []

    # minimize
    simulation.minimizeEnergy()

    # loop through number of samples
    for _ in range(100):

        # run MD for `self.n_steps_per_sample` steps
        simulation.step(1000)

        # append samples to `samples`
        samples.append(
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)
        )

    # put samples into an array
    samples = np.array(samples)

    # print out maximum deviation from center of mass
    print((samples - samples.mean(axis=(1, 2), keepdims=True)).max())

if __name__ == "__main__":
    run()
