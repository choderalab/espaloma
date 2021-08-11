# Check whether we can recover a molecular mechanics model containing just one kind of term
# Initially, interested in recovering a molecular mechanics model containing only improper torsion terms

import numpy as np
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import openmm as mm
import pytest
import espaloma as esp

import torch


def _create_impropers_only_system(
    smiles: str = "CC1=C(C(=O)C2=C(C1=O)N3CC4C(C3(C2COC(=O)N)OC)N4)N",
) -> mm.System:
    """Create a simulation that contains only improper torsion terms,
    by parameterizing with openff-1.2.0 and deleting  all terms but impropers
    """

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    g = esp.Graph(molecule)

    topology = Topology.from_molecules(molecule)
    forcefield = ForceField("openff-1.2.0.offxml")
    openmm_system = forcefield.create_openmm_system(topology)

    # delete all forces except PeriodicTorsionForce
    is_torsion = (
        lambda force: "PeriodicTorsionForce" in force.__class__.__name__
    )
    for i in range(openmm_system.getNumForces())[::-1]:
        if not is_torsion(openmm_system.getForce(i)):
            openmm_system.removeForce(i)
    assert openmm_system.getNumForces() == 1
    torsion_force = openmm_system.getForce(0)
    assert is_torsion(torsion_force)

    # set k = 0 for any torsion that's not an improper
    indices = set(
        map(
            tuple,
            esp.graphs.utils.offmol_indices.improper_torsion_indices(molecule),
        )
    )
    num_impropers_retained = 0
    for i in range(torsion_force.getNumTorsions()):
        (
            p1,
            p2,
            p3,
            p4,
            periodicity,
            phase,
            k,
        ) = torsion_force.getTorsionParameters(i)

        if (p1, p2, p3, p4) in indices:
            num_impropers_retained += 1
        else:
            torsion_force.setTorsionParameters(
                i, p1, p2, p3, p4, periodicity, phase, 0.0
            )

    assert (
        num_impropers_retained > 0
    )  # otherwise this molecule is not a useful test case!

    return openmm_system, topology, g


@pytest.mark.skip(reason="too slow")
def test_improper_recover():
    from simtk import openmm, unit
    from simtk.openmm.app import Simulation
    from simtk.unit.quantity import Quantity

    TEMPERATURE = 500 * unit.kelvin
    STEP_SIZE = 1 * unit.femtosecond
    COLLISION_RATE = 1 / unit.picosecond

    system, topology, g = _create_impropers_only_system()

    # use langevin integrator, although it's not super useful here
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE
    )

    # initialize simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator
    )

    import openff.toolkit

    # get conformer
    g.mol.generate_conformers(
        toolkit_registry=openff.toolkit.utils.RDKitToolkitWrapper(),
    )

    # put conformer in simulation
    simulation.context.setPositions(g.mol.conformers[0])

    # minimize energy
    simulation.minimizeEnergy()

    # set velocities
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)

    samples = []
    us = []

    # loop through number of samples
    for _ in range(10):

        # run MD for `self.n_steps_per_sample` steps
        simulation.step(10)

        # append samples to `samples`
        samples.append(
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(esp.units.DISTANCE_UNIT)
        )

        us.append(
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(esp.units.ENERGY_UNIT)
        )

    # put samples into an array
    samples = np.array(samples)
    us = np.array(us)

    # put samples into tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.float32)[None, :, None]

    g.heterograph.nodes["n1"].data["xyz"] = samples.permute(1, 0, 2)

    # require gradient for force matching
    g.heterograph.nodes["n1"].data["xyz"].requires_grad = True

    g.heterograph.nodes["g"].data["u_ref"] = us

    # parametrize
    layer = esp.nn.dgl_legacy.gn()
    net = torch.nn.Sequential(
        esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"]),
        esp.nn.readout.janossy.JanossyPoolingImproper(
            in_features=32, config=[32, "tanh"], out_features={"k": 6,}
        ),
        esp.mm.geometry.GeometryInGraph(),
        esp.mm.energy.EnergyInGraph(terms=["n4_improper"]),
    )

    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    for _ in range(1500):
        optimizer.zero_grad()

        net(g.heterograph)
        u_ref = g.nodes["g"].data["u"]
        u = g.nodes["g"].data["u_ref"]
        loss = torch.nn.MSELoss()(u_ref, u)
        loss.backward()
        print(loss)
        optimizer.step()

    assert loss.detach().numpy().item() < 0.1


# caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
#
#
# def _create_random_impropers_only_system(smiles: str = caffeine_smiles, k_stddev: float = 10.0) -> mm.System:
#     """Create an OpenMM system that contains only a large number of improper torsion terms,
#     assigning random coefficients ~ N(0, k_stddev) kJ/mol"""
#
#     molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
#
#     topology = Topology.from_molecules(molecule)
#     forcefield = ForceField('openff-1.2.0.offxml')
#     openmm_system = forcefield.create_openmm_system(topology)
#
#     # delete all forces
#     while openmm_system.getNumForces() > 0:
#         openmm_system.removeForce(0)
#
#     # add a torsion force
#     torsion_force = mm.PeriodicTorsionForce()
#
#     # for each improper torsion abcd, sample a periodicity, phase, and k, then add 3 terms to torsion_force
#     # with different indices abcd, acdb, adbc but identical periodicity, phase, and k
#     indices = esp.graphs.utils.offmol_indices.improper_torsion_indices(molecule)
#     improper_perms = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
#
#     for inds in indices:
#         periodicity = np.random.randint(1, 7)
#         phase = 0
#         k = np.random.randn() * k_stddev
#         for perm in improper_perms:
#             p1, p2, p3, p4 = [int(inds[p]) for p in perm]  # careful to pass python ints rather than np ints to openmm
#             torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)
#
#     openmm_system.addForce(torsion_force)
#
#     return openmm_system

# TODO: integration test where we recover this molecular mechanics system from energies/forces
