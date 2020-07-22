# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np
from simtk import openmm
from simtk import unit
from simtk.openmm.app import Simulation
from openforcefield.typing.engines.smirnoff import ForceField
from espaloma.units import *

# =============================================================================
# CONSTANTS
# =============================================================================
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
    simulation_from_graph : Create simluation from molecule.

    run : Run the simluation.

    """

    def __init__(
        self,
        forcefield='test_forcefields/smirnoff99Frosst.offxml',
        n_samples=100,
        n_steps_per_sample=1000,
        temperature=TEMPERATURE,
        collision_rate=COLLISION_RATE,
        step_size=STEP_SIZE,
    ):

        self.n_samples = n_samples
        self.n_steps_per_sample = n_steps_per_sample
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.step_size = step_size

        if isinstance(forcefield, str):
            self.forcefield = ForceField(forcefield)
        else:
            # TODO: type assertion
            self.forcefield = forcefield

    def simulation_from_graph(self, g):
        """ Create simulation from moleucle """
        # assign partial charge
        g.mol.assign_partial_charges('gasteiger') # faster

        # parameterize topology
        topology = g.mol.to_topology()

        # create openmm system
        system = self.forcefield.create_openmm_system(
            topology,

            # TODO:
            # figure out whether `sqm` should be so slow
            charge_from_molecules=[g.mol],
        )

        # use langevin integrator
        integrator = openmm.LangevinIntegrator(
            self.temperature,
            self.collision_rate,
            self.step_size
        )

        # initialize simulation
        simulation = Simulation(
            topology=topology,
            system=system,
            integrator=integrator
        )

        # get conformer
        g.mol.generate_conformers()

        # put conformer in simulation
        simulation.context.setPositions(
            g.mol.conformers[0]
        )

        # minimize energy
        simulation.minimizeEnergy()

        # set velocities
        simulation.context.setVelocitiesToTemperature(
            self.temperature
        )

        return simulation

    def run(self, g, in_place=True):
        """ Collect samples from simulation.

        Parameters
        ----------
        g : `esp.Graph`
            Input graph.

        in_place : `bool`
            If ture,

        Returns
        -------
        samples : `torch.Tensor`, `shape=(n_samples, n_nodes, 3)`
            `in_place=True`
            Sample.

        graph : `esp.Graph`
            Modified graph.

        """
        # build simulation
        simulation = self.simulation_from_graph(g)

        # initialize empty list for samples.
        samples = []

        # loop through number of samples
        for _ in range(self.n_samples):

            # run MD for `self.n_steps_per_sample` steps
            simulation.step(self.n_steps_per_sample)

            # append samples to `samples`
            samples.append(
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(DISTANCE_UNIT)
            )

        # put samples into an array
        samples = np.array(samples)

        # put samples into tensor
        samples = torch.tensor(samples, dtype=torch.float32)

        if in_place is True:
            g.heterograph.nodes['n1'].data['xyz'] = samples.permute(
                1, 0, 2
            )
            return g

        return samples
