# Check whether we can recover a molecular mechanics model containing just one kind of term
# Initially, interested in recovering a molecular mechanics model containing only improper torsion terms

import numpy as np
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm

import espaloma as esp


def _create_impropers_only_system(smiles: str = "CC1=C(C(=O)C2=C(C1=O)N3CC4C(C3(C2COC(=O)N)OC)N4)N") -> mm.System:
    """Create a simulation that contains only improper torsion terms,
    by parameterizing with openff-1.2.0 and deleting  all terms but impropers
    """

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    topology = Topology.from_molecules(molecule)
    forcefield = ForceField('openff-1.2.0.offxml')
    openmm_system = forcefield.create_openmm_system(topology)

    # delete all forces except PeriodicTorsionForce
    is_torsion = lambda force: 'PeriodicTorsionForce' in force.__class__.__name__
    for i in range(openmm_system.getNumForces())[::-1]:
        if not is_torsion(openmm_system.getForce(i)):
            openmm_system.removeForce(i)
    assert (openmm_system.getNumForces() == 1)
    torsion_force = openmm_system.getForce(0)
    assert (is_torsion(torsion_force))

    # set k = 0 for any torsion that's not an improper
    indices = set(map(tuple, esp.graphs.utils.offmol_indices.improper_torsion_indices(molecule)))
    num_impropers_retained = 0
    for i in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(i)

        if (p1, p2, p3, p4) in indices:
            num_impropers_retained += 1
        else:
            torsion_force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0.0)

    assert (num_impropers_retained > 0)  # otherwise this molecule is not a useful test case!

    return openmm_system


caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'


def _create_random_impropers_only_system(smiles: str = caffeine_smiles, k_stddev: float = 10.0) -> mm.System:
    """Create an OpenMM system that contains only a large number of improper torsion terms,
    assigning random coefficients ~ N(0, k_stddev) kJ/mol"""

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    topology = Topology.from_molecules(molecule)
    forcefield = ForceField('openff-1.2.0.offxml')
    openmm_system = forcefield.create_openmm_system(topology)

    # delete all forces
    while openmm_system.getNumForces() > 0:
        openmm_system.removeForce(0)

    # add a torsion force
    torsion_force = mm.PeriodicTorsionForce()

    # for each improper torsion abcd, sample a periodicity, phase, and k, then add 3 terms to torsion_force
    # with different indices abcd, acdb, adbc but identical periodicity, phase, and k
    indices = esp.graphs.utils.offmol_indices.improper_torsion_indices(molecule)
    improper_perms = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]

    for inds in indices:
        periodicity = np.random.randint(1, 7)
        phase = 0
        k = np.random.randn() * k_stddev
        for perm in improper_perms:
            p1, p2, p3, p4 = [int(inds[p]) for p in perm]  # careful to pass python ints rather than np ints to openmm
            torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)

    openmm_system.addForce(torsion_force)

    return openmm_system

# TODO: integration test where we recover this molecular mechanics system from energies/forces
