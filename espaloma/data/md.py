# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np
from simtk import openmm
from simtk import unit
from simtk.openmm.app import Simulation
from openforcefield.typing.engines.smirnoff import ForceField

# =============================================================================
# CONSTANTS
# =============================================================================
# units
DISTANCE_UNIT = unit.nanometer
ENERGY_UNIT = unit.kilojoule_per_mole
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT

# simulation specs
TEMPERATURE = 500 * unit.kelvin
STEP_SIZE = 1 * unit.femtosecond
COLLISION_RATE = 1 / unit.picosecond

# =============================================================================
# MODULE CLASSES
# =============================================================================
class MoleculeVacuumSimulation(object):
    """ Simluate a single molecule system in vaccum.

    Parameters
    ----------
    g : `espaloma.Graph`
        Input molecular graph.

    n_samples : `int`
        Number of samples to collect.

    n_steps_per_sample : `int`
        Number of steps between each sample.

    temperature : `float * unit.kelvin`
        Temperature for the simluation.

    collision_rate : `float / unit.picosecond`
        Collision rate.

    timestep : `float * unit.femtosecond`
        Time step.

    Methods
    -------
    sim_from_mol : Create simluation from molecule.

    """

    def __init__(
        self,
        g,
        forcefield='openff_unconstrained-1.0.0.offxml',
        n_samples=1000,
        n_steps_per_sample=1000,
        temperature=TEMPERATURE,
        collision_rate=COLLISION_RATE,
        step_size=STEP_SIZE,
    ):

        self.g = g
        self.n_samples = n_samples
        self.n_steps_per_sample = n_steps_per_sample
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.timestep = step_size

        if isinstance(forcefield, str):
            self.forcefield = ForceField(forcefield)
        else:
            # TODO: type assertion
            self.forcefield = forcefield

    def simulation_from_mol(self):
        """ Create simulation from moleucle """
        # parameterize topology
        topology = self.g.mol.to_topology()

        # create openmm system
        system = self.forcefield.create_openmm_system(topology)

        # use langevin integrator
        integrator = openmm.LangevinIntegrator(
            temperature=self.temperature,
            frictionCoeff=self.collision_rate,
            stepSize=self.step_size
        )

        # initialize simulation
        self.simulation = Simulation(
            topology=topology,
            system=system,
            integrator=integrator
        )

        # get conformer
        self.g.mol.generate_conformers()

        # put conformer in simulation
        self.simulation.context.setPositions(
            self.g.mol.conformers[0]
        )

        # minimize energy
        self.simulation.minimizeEnergy()

        # set velocities
        self.simulation.context.setVelocitiesToTemperature(
            self.temperature
        )

        return self.simulation

    def collect_samples(self):
        """ Collect samples from simulation.

        Returns
        -------
        samples : `torch.Tensor`, `shape=(n_samples, n_nodes, 3)`
        """
        # initialize empty list for samples.
        samples = []

        # loop through number of samples
        for _ in range(self.n_samples):

            # run MD for `self.n_steps_per_sample` steps
            self.simulation.step(self.n_steps_per_sample)

            # append samples to `samples`
            samples.append(
                self.simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
            )

        # put samples into an array
        samples = np.array(samples)

        # put samples into tensor
        samples = torch.tensor(samples)

        return samples
