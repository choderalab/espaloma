# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import rdkit
import torch
from openff.toolkit.typing.engines.smirnoff import ForceField
import espaloma as esp
from openmm import unit
from openmm.unit import Quantity
import math

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
OPENMM_BOND_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_LENGTH_UNIT**2)
OPENMM_ANGLE_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_ANGLE_UNIT**2)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def load_forcefield(forcefield="openff_unconstrained-2.2.1"):
    # get a forcefield
    try:
        ff = ForceField("%s.offxml" % forcefield)
    except Exception as e:
        print(e)
        raise NotImplementedError
    return ff


def openmm_system_from_graph(
    g,
    forcefield="openff_unconstrained-2.1.1",
    suffix="",
    charge_method="nn",
    create_system_kwargs={},
):
    """Construct an openmm system from `espaloma.Graph`.

    Parameters
    ----------
    g : `espaloma.Graph`
        Input graph.

    forcefield : `str`, optional, default='openff_unconstrained-2.1.1'
        Name of the force field. Have to be Open Force Field.
        (this forcefield will be used to assign nonbonded parameters, but all of its valence parameters will be overwritten)

    suffix : `str`
        Suffix for the force terms.

    charge_method : str, optional, default='nn'
        Method to use for assigning partial charges:
        'nn' : Assign partial charges from the espaloma graph net model
        'am1-bcc' : Allow the OpenFF toolkit to assign AM1-BCC charges using default backend
        'gasteiger' : Assign Gasteiger partial charges (not recommended)
        'from-molecule' : Use partial charges provided in the original `Molecule` object

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

    if charge_method == "gasteiger":
        # from rdkit.Chem.AllChem import ComputeGasteigerCharges
        # rdkit_mol = g.mol.to_rdkit()
        # ComputeGasteigerCharges(rdkit_mol)
        # charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in rdkit_mol.GetAtoms()]
        g.mol.assign_partial_charges("gasteiger")
        sys = ff.create_openmm_system(
            g.mol.to_topology(), charge_from_molecules=[g.mol]
        )

    elif charge_method == "am1-bcc":
        g.mol.assign_partial_charges("am1bcc")
        sys = ff.create_openmm_system(
            g.mol.to_topology(), charge_from_molecules=[g.mol]
        )

    elif charge_method == "from-molecule":
        sys = ff.create_openmm_system(
            g.mol.to_topology(), charge_from_molecules=[g.mol]
        )

    elif charge_method == "nn":
        g.mol.partial_charges = unit.elementary_charge * g.nodes["n1"].data[
            "q"
        ].flatten().detach().cpu().numpy().astype(
            np.float64,
        )
        sys = ff.create_openmm_system(
            g.mol.to_topology(),
            charge_from_molecules=[g.mol],
            allow_nonintegral_charges=True,
        )

    else:
        # create openmm system
        raise RuntimeError(
            "Charge method %s is not supported. " % charge_method
        )

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
                    _eq,
                    esp.units.DISTANCE_UNIT,
                ).value_in_unit(OPENMM_BOND_EQ_UNIT)

                _k = Quantity(  # bond force constant:
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

                _eq = Quantity(
                    _eq,
                    esp.units.ANGLE_UNIT,
                ).value_in_unit(OPENMM_ANGLE_EQ_UNIT)

                _k = Quantity(  # force constant
                    # since everything is enumerated twice in espaloma
                    # and once in OpenMM,
                    # we insert a coefficient of 2.0
                    _k,
                    esp.units.ANGLE_FORCE_CONSTANT_UNIT,
                ).value_in_unit(OPENMM_ANGLE_K_UNIT)

                force.setAngleParameters(idx, idx0, idx1, idx2, _eq, _k)

        if "PeriodicTorsionForce" in name:
            number_of_torsions = force.getNumTorsions()
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

                g.nodes["n4_improper"].data[
                    "periodicity%s" % suffix
                ] = torch.arange(1, 7)[None, :].repeat(
                    g.heterograph.number_of_nodes("n4_improper"), 1
                )

                g.nodes["n4_improper"].data[
                    "phases%s" % suffix
                ] = torch.zeros(
                    g.heterograph.number_of_nodes("n4_improper"), 6
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

                            if k < 0:
                                k = -k
                                _phase = math.pi - _phase

                            k = Quantity(
                                k,
                                esp.units.ENERGY_UNIT,
                            ).value_in_unit(
                                OPENMM_ENERGY_UNIT,
                            )

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
                                    k,
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
                                    k,
                                )

                            count_idx += 1

            if "k%s" % suffix in g.nodes["n4_improper"].data:
                for idx in range(
                    g.heterograph.number_of_nodes("n4_improper")
                ):
                    idx0 = g.nodes["n4_improper"].data["idxs"][idx, 0].item()
                    idx1 = g.nodes["n4_improper"].data["idxs"][idx, 1].item()
                    idx2 = g.nodes["n4_improper"].data["idxs"][idx, 2].item()
                    idx3 = g.nodes["n4_improper"].data["idxs"][idx, 3].item()

                    periodicities = g.nodes["n4_improper"].data[
                        "periodicity%s" % suffix
                    ][idx]
                    phases = g.nodes["n4_improper"].data["phases%s" % suffix][
                        idx
                    ]
                    ks = g.nodes["n4_improper"].data["k%s" % suffix][idx]
                    for sub_idx in range(ks.flatten().shape[0]):
                        k = ks[sub_idx].item()
                        if k != 0.0:
                            _periodicity = periodicities[sub_idx].item()
                            _phase = phases[sub_idx].item()

                            if k < 0:
                                k = -k
                                _phase = math.pi - _phase

                            k = Quantity(
                                k,
                                esp.units.ENERGY_UNIT,
                            ).value_in_unit(
                                OPENMM_ENERGY_UNIT,
                            )

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
                                    0.5 * k,
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
                                    0.5 * k,
                                )

                            count_idx += 1

    return sys
