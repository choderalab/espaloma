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
from os.path import exists
import tarfile

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
                [forces['vdW'][(idx, )].rmin_half._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    return g

def unbatched(num=-1):

    fname = 'parm_at_Frosst.tgz'
    url = 'http://www.ccl.net/cca/data/parm_at_Frosst/parm_at_Frosst.tgz'

    # download if we haven't already
    if not exists(fname):
        print('Downloading {} from {}...'.format(fname, url))
        import urllib.request

        urllib.request.urlretrieve(url, fname)

    # extract zinc and parm@frosst atom types
    archive = tarfile.open(fname)

    zinc_file = archive.extractfile('parm_at_Frosst/zinc.sdf')
    zinc_p_f_types_file = archive.extractfile('parm_at_Frosst/zinc_p_f_types.txt')

    zinc_p_f_types = [l.strip() for l in zinc_p_f_types_file.readlines()]

    df_sdf = Chem.ForwardSDMolSupplier(zinc_file, removeHs=False)

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

                    # g = hgfp.hierarchical_graph.from_rdkit_mol(mol)
                    g = mol_to_param_graph(mol, g)

                    ds.append(g)

                    idx += 1
                except:
                    continue

    random.shuffle(ds)

    return lambda: iter(ds)

def batched(num=-1, batch_size=32):
    return hgfp.data.utils.BatchedParamGraph(
        unbatched(num=num),
        batch_size=batch_size)
