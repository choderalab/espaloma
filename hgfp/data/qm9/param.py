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

from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')

from simtk import unit
length_unit = unit.angstrom
spring_constant_unit = unit.kilocalorie_per_mole / (unit.angstrom**2)

def mol_to_param_graph(mol, g):
    mol = Molecule.from_rdkit(mol)

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
                [forces['vdW'][(idx, )].epsilon._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    return g

def unbatched(num=-1, sdf_path='gdb9.sdf'):
    # parse data
    df_sdf = Chem.SDMolSupplier(sdf_path)

    # init
    ds = []

    idx = 0

    while True:
        try:
            mol = next(df_sdf)
        except:
            break
        if num != -1 and idx > num:
            break
        if mol != None:
            n_atoms = mol.GetNumAtoms()
            if n_atoms > 2:
                try:
                    g = hgfp.graph.from_rdkit_mol(mol)

                    g = hgfp.heterograph.from_graph(g)

                    g = mol_to_param_graph(mol, g)

                    ds.append(g)

                    idx += 1
                except:
                    continue

    random.shuffle(ds)

    return lambda: iter(ds)

def batched(num=-1, sdf_path='gdb9.sdf', batch_size=32):
    hgfp.data.utils.BatchedParamGraph(
        unbatched(num=num, sdf_path=sdf_path),
        batch_size=batch_size)
