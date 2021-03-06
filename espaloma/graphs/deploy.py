# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
import torch
from openforcefield.typing.engines.smirnoff import ForceField
import espaloma as esp
from simtk import unit
from simtk.unit.quantity import Quantity

# =============================================================================
# CONSTANTS
# =============================================================================
OPENMM_LENGTH_UNIT = unit.nanometer
OPENMM_ANGLE_UNIT = unit.radian
OPENMM_ENERGY_UNIT = unit.kilojoule_per_mole

OPENMM_BOND_EQ_UNIT = OPENMM_LENGTH_UNIT
OPENMM_ANGLE_EQ_UNIT = OPENMM_ANGLE_UNIT
OPENMM_TORSION_K_UNIT = OPENMM_ENERGY_UNIT
OPENMM_TORSION_PHASE_UNIT = OPENMM_ANGLE_UNIT
OPENMM_BOND_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_LENGTH_UNIT ** 2)
OPENMM_ANGLE_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_ANGLE_UNIT ** 2)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def load_forcefield(forcefield="openff_unconstrained-1.1.0"):
    # get a forcefield
    try:
        ff = ForceField("%s.offxml" % forcefield)
    except:
        try:
            ff = ForceField("test_forcefields/%s.offxml" % forcefield)
        except:
            raise NotImplementedError
    return ff


def openmm_system_from_graph(
    g, forcefield="openff_unconstrained-1.1.0", suffix=""
):
    """ Construct an openmm system from `espaloma.Graph`.

    Parameters
    ----------
    g : `espaloma.Graph`
        Input graph.

    forcefield : `str`
        Name of the force field. Have to be Open Force Field.
        (this forcefield will be used to assign nonbonded parameters, but all of its valence parameters will be overwritten)

    suffix : `str`
        Suffix for the force terms.

    Returns
    -------
    sys : `openmm.System`
        Constructed single-molecule OpenMM system.


    """
    ff = load_forcefield(forcefield)

    # get the mapping between position and indices
    bond_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n2"].data["idxs"])
    }

    angle_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n3"].data["idxs"])
    }

    # create openmm system
    sys = ff.create_openmm_system(g.mol.to_topology())

    for force in sys.getForces():
        name = force.__class__.__name__
        if "HarmonicBondForce" in name:
            assert force.getNumBonds() * 2 == g.heterograph.number_of_nodes(
                "n2"
            )

            for idx in range(force.getNumBonds()):
                idx0, idx1, eq, k = force.getBondParameters(idx)
                position = bond_lookup[(idx0, idx1)]
                _eq = (
                    g.nodes["n2"]
                    .data["eq%s" % suffix][position]
                    .detach()
                    .numpy()
                    .item()
                )
                _k = (
                    g.nodes["n2"]
                    .data["k%s" % suffix][position]
                    .detach()
                    .numpy()
                    .item()
                )

                _eq = Quantity(  # bond length
                    _eq, esp.units.DISTANCE_UNIT,
                ).value_in_unit(OPENMM_BOND_EQ_UNIT)

                _k = 2.0 * Quantity(  # bond force constant:
                    # since everything is enumerated twice in espaloma
                    # and once in OpenMM,
                    # we insert a coefficient of 2.0
                    _k,
                    esp.units.FORCE_CONSTANT_UNIT,
                ).value_in_unit(OPENMM_BOND_K_UNIT)

                force.setBondParameters(idx, idx0, idx1, _eq, _k)

        if "HarmonicAngleForce" in name:
            assert force.getNumAngles() * 2 == g.heterograph.number_of_nodes(
                "n3"
            )

            for idx in range(force.getNumAngles()):
                idx0, idx1, idx2, eq, k = force.getAngleParameters(idx)
                position = angle_lookup[(idx0, idx1, idx2)]
                _eq = (
                    g.nodes["n3"]
                    .data["eq%s" % suffix][position]
                    .detach()
                    .numpy()
                    .item()
                )
                _k = (
                    g.nodes["n3"]
                    .data["k%s" % suffix][position]
                    .detach()
                    .numpy()
                    .item()
                )

                _eq = Quantity(_eq, esp.units.ANGLE_UNIT,).value_in_unit(
                    OPENMM_ANGLE_EQ_UNIT
                )

                _k = 2.0 * Quantity(  # force constant
                    # since everything is enumerated twice in espaloma
                    # and once in OpenMM,
                    # we insert a coefficient of 2.0
                    _k,
                    esp.units.ANGLE_FORCE_CONSTANT_UNIT,
                ).value_in_unit(OPENMM_ANGLE_K_UNIT)

                force.setAngleParameters(idx, idx0, idx1, idx2, _eq, _k)

        if "PeriodicTorsionForce" in name:
            number_of_torsions = force.getNumTorsions()
            assert number_of_torsions <= g.heterograph.number_of_nodes("n4")

            # TODO: An alternative would be to start with an empty PeriodicTorsionForce and always call force.addTorsion

            if (
                "periodicity%s" % suffix not in g.nodes["n4"].data
                or "phase%s" % suffix not in g.nodes["n4"].data
            ):

                g.nodes["n4"].data["periodicity%s" % suffix] = torch.arange(
                    1, 7
                )[None, :].repeat(g.heterograph.number_of_nodes("n4"), 1)

                g.nodes["n4"].data["phases%s" % suffix] = torch.zeros(
                    g.heterograph.number_of_nodes("n4"), 6
                )

            count_idx = 0
            for idx in range(g.heterograph.number_of_nodes("n4")):
                idx0 = g.nodes["n4"].data["idxs"][idx, 0].item()
                idx1 = g.nodes["n4"].data["idxs"][idx, 1].item()
                idx2 = g.nodes["n4"].data["idxs"][idx, 2].item()
                idx3 = g.nodes["n4"].data["idxs"][idx, 3].item()

                # assuming both (a,b,c,d) and (d,c,b,a) are listed for every torsion, only pick one of the orderings
                if idx0 < idx3:
                    periodicities = g.nodes["n4"].data[
                        "periodicity%s" % suffix
                    ][idx]
                    phases = g.nodes["n4"].data["phases%s" % suffix][idx]
                    ks = g.nodes["n4"].data["k%s" % suffix][idx]
                    for sub_idx in range(ks.flatten().shape[0]):
                        k = ks[sub_idx].item()
                        if k != 0.0:
                            _periodicity = periodicities[sub_idx].item()
                            _phase = phases[sub_idx].item()

                            k = Quantity(
                                k, esp.units.ENERGY_UNIT,
                            ).value_in_unit(OPENMM_ENERGY_UNIT,)

                            if count_idx < number_of_torsions:
                                force.setTorsionParameters(
                                    # since everything is enumerated
                                    # twice in espaloma
                                    # and once in OpenMM,
                                    # we insert a coefficient of 2.0
                                    count_idx,
                                    idx0,
                                    idx1,
                                    idx2,
                                    idx3,
                                    _periodicity,
                                    _phase,
                                    2.0 * k,
                                )

                            else:
                                force.addTorsion(
                                    # since everything is enumerated
                                    # twice in espaloma
                                    # and once in OpenMM,
                                    # we insert a coefficient of 2.0
                                    idx0,
                                    idx1,
                                    idx2,
                                    idx3,
                                    _periodicity,
                                    _phase,
                                    2.0 * k,
                                )

                            count_idx += 1

    return sys
