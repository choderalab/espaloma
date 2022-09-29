# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import espaloma as esp
from openmm import unit
from openmm.unit import Quantity

# =============================================================================
# CONSTANTS
# =============================================================================
MOLECULES = {
    "benzene": "C1=CC=CC=C1",
    "uracil": "O=C1NC=CC(=O)N1",
    "naphthalene": "C1=CC=C2C=CC=CC2=C1",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "salicylic": "C1=CC=C(C(=C1)C(=O)O)O",
    "malonaldehyde": "C(C=O)C=O",
    "ethanol": "CCO",
    "toluene": "CC1=CC=CC=C1",
    "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "azobenzene": "C1=CC=C(C=C1)N=NC2=CC=CC=C2",
}

OFFSETS = {
    1: -0.500607632585,
    6: -37.8302333826,
    7: -54.5680045287,
    8: -75.0362229210,
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def sum_offsets(elements):
    return sum([OFFSETS[element] for element in elements])


def realize_molecule(
    data, name, smiles=None, first=-1, subtract_nonbonded=True
):
    elements = data["z"].tolist()

    offset = sum_offsets(elements)

    g = esp.data.utils.infer_mol_from_coordinates(
        data["R"][0],
        elements,
        smiles,
    )

    g.nodes["n1"].data["xyz"] = torch.tensor(
        Quantity(
            data["R"].transpose(1, 0, 2),
            unit.angstrom,
        ).value_in_unit(esp.units.DISTANCE_UNIT),
        requires_grad=True,
    )[:, :first, :]

    g.nodes["g"].data["u_ref"] = (
        torch.tensor(
            Quantity(
                data["E"],
                unit.kilocalorie_per_mole,
            ).value_in_unit(esp.units.ENERGY_UNIT)
        ).transpose(1, 0)[:, :first]
        - offset
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.tensor(
        Quantity(
            data["F"],
            unit.kilocalorie_per_mole / unit.angstrom,
        ).value_in_unit(esp.units.FORCE_UNIT)
    ).transpose(1, 0)[:, :first, :]

    if subtract_nonbonded is True:
        g = esp.data.md.subtract_nonbonded_force(g)

    return g


def get_molecule(name, *args, **kwargs):
    if name == "benzene":
        file_name = "benzene_old_dft.npz"
    else:
        file_name = "%s_dft.npz" % name

    from os.path import exists

    if not exists(file_name):
        url = "http://www.quantum-machine.org/gdml/data/npz/%s" % file_name
        print(url)
        import urllib.request

        urllib.request.urlretrieve(url, file_name)

    data = np.load(file_name)

    smiles = MOLECULES[name]

    g = realize_molecule(data, name, smiles, *args, **kwargs)

    return g
