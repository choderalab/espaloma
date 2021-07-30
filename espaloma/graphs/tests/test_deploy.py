import pytest
import numpy.testing as npt
import espaloma as esp
from simtk import openmm
from simtk import openmm as mm
from simtk import unit
import pytest
omm_angle_unit = unit.radian
omm_energy_unit = unit.kilojoule_per_mole
from simtk.unit.quantity import Quantity

from simtk.openmm import app

def test_butane():
    """check that esp.graphs.deploy.openmm_system_from_graph runs without error on butane"""
    ff = esp.graphs.legacy_force_field.LegacyForceField("openff-1.2.0")
    g = esp.Graph("CCCC")
    g = ff.parametrize(g)
    esp.graphs.deploy.openmm_system_from_graph(g, suffix="_ref")

def test_caffeine():
    ff = esp.graphs.legacy_force_field.LegacyForceField("openff-1.2.0")
    g = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    g = ff.parametrize(g)
    esp.graphs.deploy.openmm_system_from_graph(g, suffix="_ref")

def test_parameter_consistent_caffeine():
    ff = esp.graphs.legacy_force_field.LegacyForceField("openff-1.2.0")
    g = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    g = ff.parametrize(g)
    system = esp.graphs.deploy.openmm_system_from_graph(g, suffix="_ref")
    forces = list(system.getForces())
    openff_forces = ff.FF.label_molecules(g.mol.to_topology())[0]
    for idx, force in enumerate(forces):
        force.setForceGroup(idx)
        name = force.__class__.__name__
        if "HarmonicBondForce" in name:
            for _idx in range(force.getNumBonds()):
                start, end, eq, k_openmm = force.getBondParameters(_idx)

                k_openff = openff_forces["Bonds"][(start, end)].k

                npt.assert_almost_equal(
                    k_openmm / k_openff,
                    2.0,
                    decimal=3,
                )

def test_energy_consistent_caffeine():
    ff = esp.graphs.legacy_force_field.LegacyForceField("openff-1.2.0")
    g = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    g = ff.parametrize(g)
    system = esp.graphs.deploy.openmm_system_from_graph(g, suffix="_ref")

    import torch
    g.nodes['n1'].data['xyz'] = torch.randn(
        g.heterograph.number_of_nodes('n1'), 1, 3
    )
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["n2", "n3", "n4", "n4_improper"], suffix="_ref")

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
        g.mol.to_topology().to_openmm(), system, openmm.VerletIntegrator(0.0),
    )

    _simulation.context.setPositions(
        Quantity(
            g.nodes['n1'].data['xyz'][:, 0, :].numpy(),
            unit=esp.units.DISTANCE_UNIT,
        ).value_in_unit(unit.nanometer)
    )

    for idx, force in enumerate(forces):
        name = force.__class__.__name__

        state = _simulation.context.getState(
            getEnergy=True, getParameters=True, groups=2 ** idx,
        )

        energy = state.getPotentialEnergy().value_in_unit(
            esp.units.ENERGY_UNIT
        )

        energies[name] = energy

    # test bonds
    npt.assert_almost_equal(
        g.nodes["g"].data["u_n2_ref"].numpy(),
        energies["HarmonicBondForce"],
        decimal=3,
    )

    # test angles
    npt.assert_almost_equal(
        g.nodes["g"].data["u_n3_ref"].numpy(),
        energies["HarmonicAngleForce"],
        decimal=3,
    )

    npt.assert_almost_equal(
        g.nodes["g"].data["u_n4_ref"].numpy() + g.nodes["g"].data["u_n4_improper_ref"].numpy(),
        energies["PeriodicTorsionForce"],
        decimal=3,
    )



# TODO: test that desired parameters are assigned
