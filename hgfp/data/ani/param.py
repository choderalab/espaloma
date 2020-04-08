# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
from rdkit import Chem
import pandas as pd
import dgl
import torch
import os
import hgfp
import random
import h5py
import numpy as np
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')
from openeye import oechem
import tempfile
import copy


from simtk import unit
length_unit = unit.angstrom
spring_constant_unit = unit.kilocalorie_per_mole / (unit.angstrom**2)


def get_ani_mol(coordinates, species, smiles):
    """ Given smiles string and list of elements as reference,
    get the RDKit mol with xyz.

    """

    mol = oechem.OEGraphMol()

    for symbol in species:
        mol.NewAtom(getattr(oechem, 'OEElemNo_' + symbol))

    mol.SetCoords(coordinates.reshape([-1]))
    mol.SetDimension(3)
    oechem.OEDetermineConnectivity(mol)
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEPerceiveBondOrders(mol)

    smiles_can = oechem.OECreateCanSmiString(mol)

    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)
    mol_ref = next(ims.GetOEMols())
    smiles_ref = oechem.OECreateCanSmiString(mol_ref)

    assert smiles_can == smiles_ref

    g = hgfp.graph.from_oemol(mol, use_fp=True)

    return g, mol





def mol_to_param_graph(mol, g):
    mol = Molecule.from_openeye(mol)

    forces = FF.label_molecules(mol.to_topology())[0]

    g.apply_nodes(
        lambda node: {'k_ref': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy())].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'eq_ref': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy())].length._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'k_ref': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy())].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'eq_ref': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy())].angle._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'k_ref': torch.Tensor(
                [forces['vdW'][(idx, )].epsilon._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    g.apply_nodes(
        lambda node: {'eq_ref': torch.Tensor(
                [forces['vdW'][(idx, )].rmin_half._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    return g


def unbatched(num=-1, ani_path='.', *args, **kwargs):

    class _iter():

        def __iter__(self):
            idx = 0
            
            for path in os.listdir(ani_path):
                if idx > num and num != -1:
                    break
                if path.endswith('.h5'):
                    f = h5py.File(path, 'r')
                    for d0 in list(f.keys()):
                        if idx > num and num != -1:
                            break
                        for d1 in list(f[d0].keys()):
                            if idx > num and num != -1:
                                break
                            
                            try:
                                smiles = ''.join([
                                    x.decode('utf-8') for x in f[d0][d1]['smiles'].value.tolist()])
                                coordinates = f[d0][d1]['coordinates'].value
                                energies = f[d0][d1]['energies'].value
                                species = [x.decode('utf-8') for x in f[d0][d1]['species'].value]
                                

                                low_energy_idx = np.argsort(energies)[0]
                                
                                g, mol = get_ani_mol(
                                                coordinates[low_energy_idx],
                                                species,
                                                smiles)


                                g = hgfp.heterograph.from_graph(g)

                                g = mol_to_param_graph(mol, g)
                                
                                idx += 1
                                
                                yield g

                            except:
                                pass


    return _iter 


def batched(num=-1,  ani_path='.', batch_size=32):
    return hgfp.data.utils.BatchedParamGraph(
        unbatched(num=num, ani_path=ani_path),
        batch_size=batch_size)
