# =============================================================================
# IMPORTS
# =============================================================================
import hgfp
import dgl
import torch
import numpy as np
import rdkit
from rdkit import Chem
import math
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')

from simtk import unit
length_unit = unit.angstrom
spring_constant_unit = unit.kilocalorie_per_mole / (unit.angstrom**2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def u(mol, toolkit='rdkit', return_graph=False):
    if toolkit == 'rdkit':
        g = hgfp.heterograph.from_graph(hgfp.graph.from_rdkit_mol(mol))
        mol = Molecule.from_rdkit(mol)

    elif toolkit == 'openeye':
        g = hgfp.heterograph.from_graph(hgfp.graph.from_oemol(mol))
        mol = Molecule.from_openeye(mol)

    else:
        raise "Toolkit could only be either openeye or rdkit."

    forces = FF.label_molecules(mol.to_topology())[0] 

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy().astype(np.int64))].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy().astype(np.int64))].length._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy().astype(np.int64))].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy().astype(np.int64))].angle._value / 180 * math.pi for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['vdW'][(idx, )].epsilon._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['vdW'][(idx, )].rmin_half._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    if return_graph == True:
        return g

    g = hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(g)

    g = hgfp.mm.energy_in_heterograph.u(g)


    u = torch.sum(
            torch.cat(
            [
                g.nodes['mol'].data['u' + term][:, None] for term in [
                    'bond', 'angle', 'torsion', 'one_four', 'nonbonded', '0'
            ] if 'u' + term in g.nodes['mol'].data],
            dim=1),
        dim=1)
    

    return u
