# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch

from openmmforcefields.generators import SystemGenerator
from simtk import openmm, unit
from simtk.openmm.app import Simulation
from simtk.unit import Quantity

from espaloma.units import *
import espaloma as esp

# =============================================================================
# CONSTANTS
# =============================================================================
# simulation specs
TEMPERATURE = 350 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond
EPSILON_MIN = 0.05 * unit.kilojoules_per_mole

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def subtract_nonbonded_force(
    g,
    forcefield="gaff-1.81",
    subtract_charges=False,
):

    # parameterize topology
    topology = g.mol.to_topology().to_openmm()

    generator = SystemGenerator(
        small_molecule_forcefield=forcefield,
        molecules=[g.mol],
        forcefield_kwargs={"constraints": None, "removeCMMotion": False},
    )

    # create openmm system
    system = generator.create_system(
        topology,
    )

    # use langevin integrator, although it's not super useful here
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE
    )

    # create simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator
    )

    # get forces
    forces = list(system.getForces())

    # loop through forces
    for force in forces:
        name = force.__class__.__name__

        # turn off angle
        if "Angle" in name:
            for idx in range(force.getNumAngles()):
                id1, id2, id3, angle, k = force.getAngleParameters(idx)
                force.setAngleParameters(idx, id1, id2, id3, angle, 0.0)

            force.updateParametersInContext(simulation.context)

        elif "Bond" in name:
            for idx in range(force.getNumBonds()):
                id1, id2, length, k = force.getBondParameters(idx)
                force.setBondParameters(
                    idx,
                    id1,
                    id2,
                    length,
                    0.0,
                )

            force.updateParametersInContext(simulation.context)

        elif "Torsion" in name:
            for idx in range(force.getNumTorsions()):
                (
                    id1,
                    id2,
                    id3,
                    id4,
                    periodicity,
                    phase,
                    k,
                ) = force.getTorsionParameters(idx)
                force.setTorsionParameters(
                    idx,
                    id1,
                    id2,
                    id3,
                    id4,
                    periodicity,
                    phase,
                    0.0,
                )

            force.updateParametersInContext(simulation.context)

        elif "Nonbonded" in name:
            if subtract_charges:
                for idx in range(force.getNumParticles()):
                    q, sigma, epsilon = force.getParticleParameters(idx)
                    force.setParticleParameters(idx, q, sigma, epsilon)
                for idx in range(force.getNumExceptions()):
                    idx0, idx1, q, sigma, epsilon = force.getExceptionParameters(idx)
                    force.setExceptionParameters(idx, q, sigma, epsilon)

    # the snapshots
    xs = (
        Quantity(
            g.nodes["n1"].data["xyz"].detach().numpy(),
            esp.units.DISTANCE_UNIT,
        )
        .value_in_unit(unit.nanometer)
        .transpose((1, 0, 2))
    )

    # loop through the snapshots
    energies = []
    derivatives = []

    for x in xs:
        simulation.context.setPositions(x)

        state = simulation.context.getState(
            getEnergy=True,
            getParameters=True,
            getForces=True,
        )

        energy = state.getPotentialEnergy().value_in_unit(
            esp.units.ENERGY_UNIT,
        )

        derivative = state.getForces(asNumpy=True).value_in_unit(
            esp.units.FORCE_UNIT,
        )

        energies.append(energy)
        derivatives.append(derivative)

    # put energies to a tensor
    energies = torch.tensor(
        energies,
        dtype=torch.get_default_dtype(),
    ).flatten()[None, :]
    derivatives = torch.tensor(
        np.stack(derivatives, axis=1),
        dtype=torch.get_default_dtype(),
    )

    # subtract the energies
    g.heterograph.apply_nodes(
        lambda node: {"u_ref": node.data["u_ref"] - energies},
        ntype="g",
    )

    if "u_ref_prime" in g.nodes["n1"]:
        g.heterograph.apply_nodes(
            lambda node: {
                "u_ref_prime": node.data["u_ref_prime"] - derivatives
            },
            ntype="n1",
        )

    return g


def subtract_nonbonded_force_except_14(
    g,
    forcefield="gaff-1.81",
):

    # parameterize topology
    topology = g.mol.to_topology().to_openmm()

    generator = SystemGenerator(
        small_molecule_forcefield=forcefield,
        molecules=[g.mol],
    )

    # create openmm system
    system = generator.create_system(
        topology,
    )

    # use langevin integrator, although it's not super useful here
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE
    )

    # create simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator
    )

    # get forces
    forces = list(system.getForces())

    # loop through forces
    for force in forces:
        name = force.__class__.__name__

        # turn off angle
        if "Angle" in name:
            for idx in range(force.getNumAngles()):
                id1, id2, id3, angle, k = force.getAngleParameters(idx)
                force.setAngleParameters(idx, id1, id2, id3, angle, 0.0)

            force.updateParametersInContext(simulation.context)

        elif "Bond" in name:
            for idx in range(force.getNumBonds()):
                id1, id2, length, k = force.getBondParameters(idx)
                force.setBondParameters(
                    idx,
                    id1,
                    id2,
                    length,
                    0.0,
                )

            force.updateParametersInContext(simulation.context)

        elif "Torsion" in name:
            for idx in range(force.getNumTorsions()):
                (
                    id1,
                    id2,
                    id3,
                    id4,
                    periodicity,
                    phase,
                    k,
                ) = force.getTorsionParameters(idx)
                force.setTorsionParameters(
                    idx,
                    id1,
                    id2,
                    id3,
                    id4,
                    periodicity,
                    phase,
                    0.0,
                )

            force.updateParametersInContext(simulation.context)

        elif "Nonbonded" in name:
            for exception_index in range(force.getNumExceptions()):
                (
                    p1,
                    p2,
                    chargeprod,
                    sigma,
                    epsilon,
                ) = force.getExceptionParameters(exception_index)
                force.setExceptionParameters(
                    exception_index, p1, p2, chargeprod, sigma, 1e-8 * epsilon
                )

            force.updateParametersInContext(simulation.context)

    # the snapshots
    xs = (
        Quantity(
            g.nodes["n1"].data["xyz"].detach().numpy(),
            esp.units.DISTANCE_UNIT,
        )
        .value_in_unit(unit.nanometer)
        .transpose((1, 0, 2))
    )

    # loop through the snapshots
    energies = []
    derivatives = []

    for x in xs:
        simulation.context.setPositions(x)

        state = simulation.context.getState(
            getEnergy=True,
            getParameters=True,
            getForces=True,
        )

        energy = state.getPotentialEnergy().value_in_unit(
            esp.units.ENERGY_UNIT,
        )

        derivative = state.getForces(asNumpy=True).value_in_unit(
            esp.units.FORCE_UNIT,
        )

        energies.append(energy)
        derivatives.append(derivative)

    # put energies to a tensor
    energies = torch.tensor(
        energies,
        dtype=torch.get_default_dtype(),
    ).flatten()[None, :]
    derivatives = torch.tensor(
        np.stack(derivatives, axis=1),
        dtype=torch.get_default_dtype(),
    )

    # subtract the energies
    g.heterograph.apply_nodes(
        lambda node: {"u_ref": node.data["u_ref"] - energies},
        ntype="g",
    )

    if "u_ref_prime" in g.nodes["n1"].data:

        g.heterograph.apply_nodes(
            lambda node: {
                "u_ref_prime": node.data["u_ref_prime"] - derivatives
            },
            ntype="n1",
        )

    return g


# =============================================================================
# MODULE CLASSES
# =============================================================================
class MoleculeVacuumSimulation(object):
    """Simluate a single molecule system in vaccum.

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
        forcefield="gaff-1.81",
        n_samples=100,
        n_conformers=10,
        n_steps_per_sample=1000,
        temperature=TEMPERATURE,
        collision_rate=COLLISION_RATE,
        step_size=STEP_SIZE,
        charge_method=None,
    ):

        self.n_samples = n_samples
        self.n_steps_per_sample = n_steps_per_sample
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.step_size = step_size
        self.forcefield = forcefield
        self.n_conformers = n_conformers
        self.charge_method = charge_method

    def simulation_from_graph(self, g):
        """ Create simulation from moleucle """
        # assign partial charge
        if self.charge_method is not None:
            g.mol.assign_partial_charges(self.charge_method)

        # parameterize topology
        topology = g.mol.to_topology().to_openmm()

        generator = SystemGenerator(
            small_molecule_forcefield=self.forcefield,
            molecules=[g.mol],
        )

        # create openmm system
        system = generator.create_system(
            topology,
        )

        # set epsilon minimum to 0.05 kJ/mol
        for force in system.getForces():
            if "Nonbonded" in force.__class__.__name__:
                for particle_index in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(
                        particle_index
                    )
                    if epsilon < EPSILON_MIN:
                        force.setParticleParameters(
                            particle_index, charge, sigma, EPSILON_MIN
                        )

        # use langevin integrator
        integrator = openmm.LangevinIntegrator(
            self.temperature, self.collision_rate, self.step_size
        )

        # initialize simulation
        simulation = Simulation(
            topology=topology,
            system=system,
            integrator=integrator,
            platform=openmm.Platform.getPlatformByName("Reference"),
        )

        return simulation

    def run(self, g, in_place=True):
        """Collect samples from simulation.

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

        import openff.toolkit

        # get conformer
        g.mol.generate_conformers(
            toolkit_registry=openff.toolkit.utils.RDKitToolkitWrapper(),
            n_conformers=self.n_conformers,
        )

        # get number of actual conformers
        true_n_conformers = len(g.mol.conformers)

        samples = []
        for idx in range(true_n_conformers):
            # put conformer in simulation
            simulation.context.setPositions(g.mol.conformers[idx])

            # set velocities
            simulation.context.setVelocitiesToTemperature(self.temperature)

            # minimize
            simulation.minimizeEnergy()

            # loop through number of samples
            for _ in range(self.n_samples // self.n_conformers):

                # run MD for `self.n_steps_per_sample` steps
                simulation.step(self.n_steps_per_sample)

                # append samples to `samples`
                samples.append(
                    simulation.context.getState(getPositions=True)
                    .getPositions(asNumpy=True)
                    .value_in_unit(DISTANCE_UNIT)
                )

        # if the `samples` array is not filled,
        # pick a random conformer to do it again
        if len(samples) < self.n_samples:
            len_samples = len(samples)
            import random

            idx = random.choice(list(range(true_n_conformers)))
            simulation.context.setPositions(g.mol.conformers[idx])

            # set velocities
            simulation.context.setVelocitiesToTemperature(self.temperature)

            # minimize
            simulation.minimizeEnergy()

            # loop through number of samples
            for _ in range(self.n_samples - len_samples):

                # run MD for `self.n_steps_per_sample` steps
                simulation.step(self.n_steps_per_sample)

                # append samples to `samples`
                samples.append(
                    simulation.context.getState(getPositions=True)
                    .getPositions(asNumpy=True)
                    .value_in_unit(DISTANCE_UNIT)
                )

        print(samples)
        assert len(samples) == self.n_samples

        # put samples into an array
        samples = np.array(samples)

        # put samples into tensor
        samples = torch.tensor(samples, dtype=torch.float32)

        if in_place is True:
            g.heterograph.nodes["n1"].data["xyz"] = samples.permute(1, 0, 2)

            # require gradient for force matching
            g.heterograph.nodes["n1"].data["xyz"].requires_grad = True

            return g

        return samples
