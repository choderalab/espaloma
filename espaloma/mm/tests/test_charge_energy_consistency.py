import pytest
import espaloma as esp
import numpy as np
import numpy.testing as npt
import pytest
import torch
from simtk import openmm
from simtk import openmm as mm
from simtk import unit

@pytest.mark.parametrize(
    "g",
    esp.data.esol(first=5), # use a subset of ESOL dataset to test
)
def test_coulomb_energy_consistency(g):
    """ We use both `esp.mm` and OpenMM to compute the Coulomb energy of
    some molecules with generated geometries and see if the resulting Columb
    energy matches.
    

    """
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    # get simulation
    esp_simulation = MoleculeVacuumSimulation(
        n_samples=1,
        n_steps_per_sample=10,
        forcefield="gaff-1.81",
        charge_method="gasteiger",
    )

    simulation = esp_simulation.simulation_from_graph(g)
    charges = g.mol.partial_charges.flatten()
    system = simulation.system
    esp_simulation.run(g, in_place=True)

    # if MD blows up, forget about it
    if g.nodes["n1"].data["xyz"].abs().max() > 100:
        return True

    _simulation = openmm.app.Simulation(
        simulation.topology,
        system,
        openmm.VerletIntegrator(0.0),
    )

    forces = list(system.getForces())
    for force in forces:
        name = force.__class__.__name__
        if "Nonbonded" in name:
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            force.updateParametersInContext(_simulation.context)

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
        if "Nonbonded" in name:
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

    g.nodes['n1'].data['q'] = torch.tensor(charges).unsqueeze(-1)
    esp.mm.nonbonded.multiply_charges(g.heterograph)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["nonbonded", "onefour"])

    npt.assert_almost_equal(
        g.nodes['g'].data['u'].item(),
        energy_old - energy_new,
        decimal=3,
    )
