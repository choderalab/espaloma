import numpy as np
import numpy.testing as npt
import pytest
import torch
from simtk import openmm, unit

import espaloma as esp
from espaloma.units import *


@pytest.mark.parametrize(
    "g", esp.data.esol(first=10),
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

        name = force.__class__.__name__

        if "Nonbonded" in name:
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

            # epsilons = {}
            # sigmas = {}

            # for _idx in range(force.getNumParticles()):
            #     q, sigma, epsilon = force.getParticleParameters(_idx)

            #     # record parameters
            #     epsilons[_idx] = epsilon
            #     sigmas[_idx] = sigma

            #     force.setParticleParameters(_idx, 0., sigma, epsilon)

            # def sigma_combining_rule(sig1, sig2):
            #     return (sig1 + sig2) / 2

            # def eps_combining_rule(eps1, eps2):
            #     return np.sqrt(np.abs(eps1 * eps2))

            # for _idx in range(force.getNumExceptions()):
            #     idx0, idx1, q, sigma, epsilon = force.getExceptionParameters(
            #         _idx)
            #     force.setExceptionParameters(
            #         _idx,
            #         idx0,
            #         idx1,
            #         0.0,
            #         sigma_combining_rule(sigmas[idx0], sigmas[idx1]),
            #         eps_combining_rule(epsilons[idx0], epsilons[idx1])
            #     )

            # force.updateParametersInContext(_simulation.context)

    # create new simulation
    _simulation = openmm.app.Simulation(
        simulation.topology, system, openmm.VerletIntegrator(0.0),
    )

    _simulation.context.setPositions(
        simulation.context.getState(getPositions=True).getPositions()
    )

    for idx, force in enumerate(forces):

        name = force.__class__.__name__

        state = _simulation.context.getState(
            getEnergy=True, getParameters=True, groups=2 ** idx,
        )

        energy = state.getPotentialEnergy().value_in_unit(ENERGY_UNIT)

        energies[name] = energy

    # parametrize
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    g = ff.parametrize(g)

    # n2 : bond, n3: angle, n1: nonbonded?
    # n1 : sigma (k), epsilon (eq), and charge (not included yet)
    for term in ["n2", "n3"]:
        g.nodes[term].data["k"] = g.nodes[term].data["k_ref"]
        g.nodes[term].data["eq"] = g.nodes[term].data["eq_ref"]

    for term in ["n1"]:
        g.nodes[term].data["sigma"] = g.nodes[term].data["sigma_ref"]
        g.nodes[term].data["epsilon"] = g.nodes[term].data["epsilon_ref"]
        # g.nodes[term].data['q'] = g.nodes[term].data['q_ref']

    for term in ["n4"]:
        g.nodes[term].data["phases"] = g.nodes[term].data["phases_ref"]
        g.nodes[term].data["periodicity"] = g.nodes[term].data["periodicity_ref"]
        g.nodes[term].data["k"] = g.nodes[term].data["k_ref"]

    # for each atom, store n_snapshots x 3
    g.nodes["n1"].data["xyz"] = torch.tensor(
        simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(DISTANCE_UNIT),
        dtype=torch.float32,
    )[None, :, :].permute(1, 0, 2)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph)
    # writes into nodes
    # .data['u_nonbonded'], .data['u_onefour'], .data['u2'], .data['u3'],

    # test bonds
    npt.assert_almost_equal(
        g.nodes["g"].data["u_n2"].numpy(),
        energies["HarmonicBondForce"],
        decimal=3,
    )

    # test angles
    npt.assert_almost_equal(
        g.nodes["g"].data["u_n3"].numpy(),
        energies["HarmonicAngleForce"],
        decimal=3,
    )

    npt.assert_almost_equal(
        g.nodes["g"].data["u_n4"].numpy(),
        energies["PeriodicTorsionForce"],
        decimal=1,
    )

    print(g.nodes["g"].data["u_n4"].numpy())
    print(energies["PeriodicTorsionForce"])

    # TODO:
    # This is not working now, matching OpenMM nonbonded.
    # test nonbonded
    # TODO: must set all charges to zero in _simulation for this to pass currently, since g doesn't have any charges
    # npt.assert_almost_equal(
    #     g.nodes['g'].data['u_nonbonded'].numpy()\
    #     + g.nodes['g'].data['u_onefour'].numpy(),
    #     energies['NonbondedForce'],
    #     decimal=3,
    # )
