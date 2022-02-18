import pytest
import espaloma as esp
import numpy as np
import numpy.testing as npt
import pytest
import torch
from simtk import openmm
from simtk import openmm as mm
from simtk import unit

def test_coulomb_energy_consistency():
    """ We use both `esp.mm` and OpenMM to compute the Coulomb energy of
    some molecules with generated geometries and see if the resulting Columb
    energy matches.


    """

    g = esp.Graph("C1=CC=C1")

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

    g.mol._partial_charges = _partial_charges * unit.elementary_charge
    g.nodes['n1'].data['xyz'] = torch.tensor(_conformers).unsqueeze(-2)
    g.nodes['n1'].data['q'] = torch.tensor(_partial_charges).unsqueeze(-1)

    # parameterize topology
    topology = g.mol.to_topology().to_openmm()

    from openmmforcefields.generators import SystemGenerator
    generator = SystemGenerator(
        small_molecule_forcefield="gaff-1.81",
        molecules=[g.mol],
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
        if "Nonbonded" in name:
            print("hey")
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            force.updateParametersInContext(_simulation.context)
            for idx in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(idx)
                print(q)

    _simulation.context.setPositions(
        g.nodes["n1"].data["xyz"][:, 0, :].detach().numpy() * unit.bohr
    )

    state = _simulation.context.getState(
        getEnergy=True,
        getParameters=True,
    )

    energy_old = state.getPotentialEnergy().value_in_unit(
        esp.units.ENERGY_UNIT
    )

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

    energy_new = state.getPotentialEnergy().value_in_unit(
        esp.units.ENERGY_UNIT
    )

    esp.mm.nonbonded.multiply_charges(g.heterograph)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["nonbonded", "onefour"])

    print(g.nodes['onefour'].data['idxs'])

    print(energy_old - energy_new)
    print(g.nodes['g'].data['u'])
    
    npt.assert_almost_equal(
        g.nodes['g'].data['u'].item(),
        energy_old - energy_new,
        decimal=3,
    )

test_coulomb_energy_consistency()
