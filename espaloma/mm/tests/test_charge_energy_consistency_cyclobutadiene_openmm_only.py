import pytest
import numpy as np
from simtk import openmm
from simtk import unit

def test_coulomb_energy_consistency_cyclobutadiene():
    from openff.toolkit.topology import Molecule
    mol = Molecule.from_smiles("C1=CC=C1")

    _partial_charges = np.concatenate(
        [
            -np.ones(4),
            np.ones(4),
        ]
    )

    _conformers = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [-2.0, 2.0, 0.0]
        ]
    )

    mol._partial_charges = _partial_charges * unit.elementary_charge

    # parameterize topology
    topology = mol.to_topology().to_openmm()

    from openmmforcefields.generators import SystemGenerator
    generator = SystemGenerator(
        small_molecule_forcefield="gaff-1.81",
        molecules=[mol],
    )

    # create openmm system
    system = generator.create_system(
        topology,
    )

    _simulation = openmm.app.Simulation(
        topology,
        system,
        openmm.VerletIntegrator(0.0),
    )

    forces = list(system.getForces())
    for force in forces:
        name = force.__class__.__name__
        if name == "NonbondedForce":
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            force.updateParametersInContext(_simulation.context)
            for idx in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(idx)
                print(q)

    _simulation.context.setPositions(
        _conformers * unit.bohr
    )

    state = _simulation.context.getState(
        getEnergy=True,
        getParameters=True,
    )

    energy_old = state.getPotentialEnergy()

    forces = list(system.getForces())

    for force in forces:
        name = force.__class__.__name__
        if name == "NonbondedForce":
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            for idx in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(idx)
                force.setParticleParameters(idx, 0.0, sigma, epsilon)

            for idx in range(force.getNumExceptions()):
                idx0, idx1, q, sigma, epsilon = force.getExceptionParameters(idx)
                force.setExceptionParameters(idx, idx0, idx1, 0.0, sigma, epsilon)

            force.updateParametersInContext(_simulation.context)

    state = _simulation.context.getState(
        getEnergy=True,
        getParameters=True,
    )

    energy_new = state.getPotentialEnergy()
    print(energy_old - energy_new)
