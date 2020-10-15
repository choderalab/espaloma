# Check whether we can recover a molecular mechanics model containing just one kind of term
# Initially, interested in recovering a molecular mechanics model containing only improper torsion terms

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm

import espaloma as esp


def _create_impropers_only_system(smiles: str = "CC1=C(C(=O)C2=C(C1=O)N3CC4C(C3(C2COC(=O)N)OC)N4)N") -> mm.System:
    """Create a simulation that contains only contains improper torsion terms"""

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

# TODO: integration test where we recover this molecular mechanics system from energies/forces
