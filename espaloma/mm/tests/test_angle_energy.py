import numpy as np
import numpy.testing as npt
import pytest
import torch
import openmm
from openmm import unit

from espaloma.utils.geometry import _sample_four_particle_torsion_scan

omm_angle_unit = unit.radian
omm_energy_unit = unit.kilojoule_per_mole

from openmm import app

import espaloma as esp


def test_energy_angle_and_bond():
    g = esp.Graph("C")
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    # get simulation
    esp_simulation = MoleculeVacuumSimulation(
        n_samples=1, n_steps_per_sample=10, forcefield="gaff-1.81"
    )

    simulation = esp_simulation.simulation_from_graph(g)
    system = simulation.system
    esp_simulation.run(g)

    forces = list(system.getForces())

    energies = {}

    for idx, force in enumerate(forces):
        force.setForceGroup(idx)

        name = force.__class__.__name__

        if "Nonbonded" in name:
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    # create new simulation
    _simulation = openmm.app.Simulation(
        simulation.topology,
        system,
        openmm.VerletIntegrator(0.0),
    )

    _simulation.context.setPositions(
        g.nodes["n1"].data["xyz"][:, 0, :].detach().numpy() * unit.nanometer
    )

    for idx, force in enumerate(forces):
        name = force.__class__.__name__

        state = _simulation.context.getState(
            getEnergy=True,
            getParameters=True,
            groups=2**idx,
        )

        energy = state.getPotentialEnergy().value_in_unit(
            esp.units.ENERGY_UNIT
        )

        energies[name] = energy

    for idx, force in enumerate(forces):
        name = force.__class__.__name__
        if "HarmonicAngleForce" in name:
            print("openmm thinks there are %s angles" % force.getNumAngles())

            for _idx in range(force.getNumAngles()):
                _, __, ___, eq, k = force.getAngleParameters(_idx)
                eq = eq.value_in_unit(esp.units.ANGLE_UNIT)
                k = k.value_in_unit(esp.units.ANGLE_FORCE_CONSTANT_UNIT)
                print(eq, k)

    # parametrize
    ff = esp.graphs.legacy_force_field.LegacyForceField("gaff-1.81")
    g = ff.parametrize(g)

    # n2 : bond, n3: angle, n1: nonbonded?
    # n1 : sigma (k), epsilon (eq), and charge (not included yet)
    for term in ["n2", "n3"]:
        g.nodes[term].data["k"] = g.nodes[term].data["k_ref"]
        g.nodes[term].data["eq"] = g.nodes[term].data["eq_ref"]

    print(
        "espaloma thinks there are %s angles"
        % g.heterograph.number_of_nodes("n3")
    )
    print(g.nodes["n3"].data["k"])
    print(g.nodes["n3"].data["eq"])

    # for each atom, store n_snapshots x 3
    # g.nodes["n1"].data["xyz"] = torch.tensor(
    #     simulation.context.getState(getPositions=True)
    #     .getPositions(asNumpy=True)
    #     .value_in_unit(esp.units.DISTANCE_UNIT),
    #     dtype=torch.float32,
    # )[None, :, :].permute(1, 0, 2)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["n2", "n3", "n4"])

    n_decimals = 3

    # test angles
    npt.assert_almost_equal(
        g.nodes["g"].data["u_n3"].detach().numpy(),
        energies["HarmonicAngleForce"],
        decimal=n_decimals,
    )


if __name__ == "__main__":
    test_energy_angle_and_bond()
