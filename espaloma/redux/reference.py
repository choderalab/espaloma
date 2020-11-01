from openforcefield.typing.engines.smirnoff import ForceField

forcefield = ForceField('openff-1.2.0.offxml')

from .transforms import null_params_from_offmol
from .symmetry import offmol_to_indices
import espaloma.units as esp_units
from simtk import unit


def get_ref_params(offmol):
    labeled_mol = forcefield.label_molecules(offmol.to_topology())[0]
    params = null_params_from_offmol(offmol)
    inds = offmol_to_indices(offmol)

    # TODO: nonbonded
    set_bonds(labeled_mol, inds, params)
    set_angles(labeled_mol, inds, params)
    set_propers(labeled_mol, inds, params)
    set_impropers(labeled_mol, inds, params)

    return params


def set_bonds(labeled_mol, inds, params):
    for i, key in enumerate(inds.bonds):
        bond_info = labeled_mol['Bonds'][key]
        params.bonds[i, 0] = bond_info.k.value_in_unit(
            esp_units.FORCE_CONSTANT_UNIT)
        params.bonds[i, 1] = bond_info.length.value_in_unit(
            esp_units.DISTANCE_UNIT)


def set_angles(labeled_mol, inds, params):
    for i, key in enumerate(inds.angles):
        angle_info = labeled_mol['Angles'][key]
        params.angles[i, 0] = angle_info.k.value_in_unit(
            esp_units.ANGLE_FORCE_CONSTANT_UNIT)
        params.angles[i, 1] = angle_info.angle.value_in_unit(
            esp_units.ANGLE_UNIT)


def set_propers(labeled_mol, inds, params):
    for i, key in enumerate(inds.propers):
        proper_info = labeled_mol['ProperTorsions'][key]

        for j, period in enumerate(proper_info.periodicity):

            phase = proper_info.phase[j]
            if phase == (180 * unit.degree):
                sign = -1
            elif phase == (0 * unit.degree):
                sign = +1
            else:
                print(
                    'warning: failed assumption that phase in {0, 180} degrees')
                sign = +1

            k = proper_info.k[j].value_in_unit(esp_units.ENERGY_UNIT)
            params.propers[i, period - 1] = sign * k


def set_impropers(labeled_mol, inds, params):
    for i, key in enumerate(inds.impropers):
        improper_info = labeled_mol['ImproperTorsions'][key]

        for j, period in enumerate(improper_info.periodicity):

            phase = improper_info.phase[j]
            if phase == (180 * unit.degree):
                sign = -1
            elif phase == (0 * unit.degree):
                sign = +1
            else:
                print(
                    'warning: failed assumption that phase in {0, 180} degrees')
                sign = +1

            k = improper_info.k[j].value_in_unit(esp_units.ENERGY_UNIT)
            params.impropers[i, period - 1] = sign * k
