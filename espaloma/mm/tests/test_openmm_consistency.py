import pytest
import espaloma as esp
from simtk import openmm
from simtk import unit
import torch
from espaloma.units import *

import numpy.testing as npt

@pytest.mark.parametrize(
    "g",
    esp.data.esol(first=20),
)
def test_energy_angle_and_bond(g):
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    # get simulation
    simulation = MoleculeVacuumSimulation(
        n_samples=1, n_steps_per_sample=10
    ).simulation_from_graph(g)

    system = simulation.system

    forces = list(system.getForces())

    energies = {}

    for idx, force in enumerate(forces):
        force.setForceGroup(idx)

    # create new simulation
    _simulation = openmm.app.Simulation(
        simulation.topology,
        system,
        openmm.VerletIntegrator(0.0),
    )

    _simulation.context.setPositions(
        simulation.context.getState(getPositions=True)
            .getPositions()
    )

    for idx, force in enumerate(forces):

        state = _simulation.context.getState(
            getEnergy=True,
            getParameters=True,
            groups=2**idx,
        )

        name = force.__class__.__name__

        energy = state.getPotentialEnergy().value_in_unit(ENERGY_UNIT)

        energies[name] = energy

    # parametrize
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    g = ff.parametrize(g)

    for term in ['n2', 'n3']:
        g.nodes[term].data['k'] = g.nodes[term].data['k_ref']
        g.nodes[term].data['eq'] = g.nodes[term].data['eq_ref']

    g.nodes['n1'].data['xyz'] = torch.tensor(
        simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(DISTANCE_UNIT),
            dtype=torch.float32)[None, :, :].permute(1, 0, 2)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph)

    npt.assert_almost_equal(
        g.nodes['g'].data['u2'].numpy(),
        energies['HarmonicBondForce'],
        decimal=3,
    )

    npt.assert_almost_equal(
        g.nodes['g'].data['u3'].numpy(),
        energies['HarmonicAngleForce'],
        decimal=3,
    )
