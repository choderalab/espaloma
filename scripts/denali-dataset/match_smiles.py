import pickle
import torch
import espaloma as esp
import numpy as np
from openeye import oechem


def run(path, u_thres=0.1):
    denali_data = pickle.load(open(path, "rb"))

    for key in denali_data.keys():
        try:
            if key.split("_")[-1] == "conformers":
                if 'canonical_smiles' not in denali_data[key].keys():
                    continue
                smiles = denali_data[key]['canonical_smiles']
                xs = denali_data[key]['coordinates']
                us = denali_data[key]['energies']
                species = denali_data[key]['species']
                
                idxs = list(range(len(xs)))
                idx_ref = us.argmin()
                ok_idxs = [idx for idx in idxs if us[idx] <= us[idx_ref] + u_thres]


                g = infer_mol_from_coordinates(xs[idx_ref], species, smiles_ref=smiles)

                final_idxs = [idx_ref]
                for idx in ok_idxs:
                    if idx == idx_ref:
                        continue
                    if check_offeq_graph(xs[idx], species, smiles_ref=smiles):
                        final_idxs.append(idx)

                g.nodes['n1'].data['xyz'] = torch.tensor(xs[final_idxs, :, :]).transpose(1, 0)
                g.nodes['g'].data['u_ref'] = torch.tensor(us[None, final_idxs])
                g.save("denali/%s" % (key))
        except Exception as ex:
            print(ex)
            with open("./denali_smiles_errors.dat", "a") as error_file:
                error_file.write(f"{key}, {smiles}, {ex}\n")


def infer_mol_from_coordinates(
    coordinates,
    species,
    smiles_ref=None,
    coordinates_unit="angstrom",
):

    # local import
    from simtk import unit
    from simtk.unit import Quantity

    if isinstance(coordinates_unit, str):
        coordinates_unit = getattr(unit, coordinates_unit)

    # make sure we have the coordinates
    # in the unit system
    coordinates = Quantity(coordinates, coordinates_unit).value_in_unit(
        unit.angstrom  # to make openeye happy
    )

    # initialize molecule
    mol = oechem.OEGraphMol()

    if all(isinstance(symbol, str) for symbol in species):
        [
            mol.NewAtom(getattr(oechem, "OEElemNo_" + symbol))
            for symbol in species
        ]

    elif all(isinstance(symbol, int) for symbol in species):
        [
            mol.NewAtom(
                getattr(
                    oechem, "OEElemNo_" + oechem.OEGetAtomicSymbol(symbol)
                )
            )
            for symbol in species
        ]

    else:
        raise RuntimeError(
            "The species can only be all strings or all integers."
        )

    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles_ref)
    ref_mol = next(ims.GetOEMols())

    mol.SetCoords(coordinates.reshape([-1]))
    mol.SetDimension(3)
    oechem.OEDetermineConnectivity(mol)
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEPerceiveBondOrders(mol)

    smiles_can = oechem.OEMolToSmiles(mol)
    smiles_ref = oechem.OEMolToSmiles(mol)
    if smiles_ref != smiles_can:
        print([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        print([atom.GetAtomicNum() for atom in ref_mol.GetAtoms()])
        tmp_mol = oechem.OEGraphMol()
        if all(isinstance(symbol, str) for symbol in species):
            [
                tmp_mol.NewAtom(getattr(oechem, "OEElemNo_" + symbol))
                for symbol in species
            ]

        elif all(isinstance(symbol, int) for symbol in species):
            [
                tmp_mol.NewAtom(
                    getattr(
                        oechem, "OEElemNo_" + oechem.OEGetAtomicSymbol(symbol)
                    )
                )
                for symbol in species
            ]
        tmp_mol.SetCoords(coordinates.reshape([-1]))
        tmp_mol.SetDimension(3)
        print([(atom.GetAtomicNum(), atom.GetValence()) for atom in tmp_mol.GetAtoms()])
        oechem.OEDetermineConnectivity(tmp_mol)
        print([(atom.GetAtomicNum(), atom.GetValence()) for atom in tmp_mol.GetAtoms()])
        oechem.OEFindRingAtomsAndBonds(tmp_mol)
        print([(atom.GetAtomicNum(), atom.GetValence()) for atom in tmp_mol.GetAtoms()])
        oechem.OEPerceiveBondOrders(tmp_mol)
        print([(atom.GetAtomicNum(), atom.GetValence()) for atom in tmp_mol.GetAtoms()])
        print([oechem.OECheckAtomValence(atom) for atom in tmp_mol.GetAtoms()])

        assert (
            smiles_ref == smiles_can
        ), "SMILES different. Input is %s, ref is %s" % (
            smiles_can,
            smiles_ref,
        )

    from openff.toolkit.topology import Molecule

    _mol = Molecule.from_openeye(mol, allow_undefined_stereo=True)
    g = esp.Graph(_mol)

    return g


def check_offeq_graph(
    coordinates,
    species,
    smiles_ref=None,
    coordinates_unit="angstrom",
):

    # local import
    from simtk import unit
    from simtk.unit import Quantity

    if isinstance(coordinates_unit, str):
        coordinates_unit = getattr(unit, coordinates_unit)

    # make sure we have the coordinates
    # in the unit system
    coordinates = Quantity(coordinates, coordinates_unit).value_in_unit(
        unit.angstrom  # to make openeye happy
    )

    # initialize molecule
    mol = oechem.OEGraphMol()

    if all(isinstance(symbol, str) for symbol in species):
        [
            mol.NewAtom(getattr(oechem, "OEElemNo_" + symbol))
            for symbol in species
        ]

    elif all(isinstance(symbol, int) for symbol in species):
        [
            mol.NewAtom(
                getattr(
                    oechem, "OEElemNo_" + oechem.OEGetAtomicSymbol(symbol)
                )
            )
            for symbol in species
        ]

    else:
        raise RuntimeError(
            "The species can only be all strings or all integers."
        )

    mol.SetCoords(coordinates.reshape([-1]))
    mol.SetDimension(3)
    oechem.OEDetermineConnectivity(mol)
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEPerceiveBondOrders(mol)

    smiles_can = oechem.OEMolToSmiles(mol)
    smiles_ref = oechem.OEMolToSmiles(mol)
    if smiles_ref != smiles_can:
        return False
    return True


if __name__ == "__main__":
    import sys
    run(sys.argv[1])