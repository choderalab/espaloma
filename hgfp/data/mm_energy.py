# =============================================================================
# IMPORTS
# =============================================================================
import hgfp
import dgl
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def u(mol, toolkit='rdkit'):
    if tookit == 'rdkit':
        g = hgfp.heterograph.from_graph(hgfp.graph.from_rdkit_mol)
        mol = Molecule.from_rdkit(mol)

    elif toolkit == 'openeye':
        g = hgfp.heterograph.from_graph(hgfp.graph.from_oemol(mol))
        mol = Molecule.from_openeye(mol)

    else:
        raise "Toolkit could only be either openeye or rdkit."

    forces = FF.label_molecules(mol.to_topology())[0]

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy())].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['Bonds'][tuple(node.data['idxs'][idx].numpy())].length._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='bond')

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy())].k._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['Angles'][tuple(node.data['idxs'][idx].numpy())].angle._value for idx in range(node.data['idxs'].shape[0])])},
        ntype='angle')

    g.apply_nodes(
        lambda node: {'k': torch.Tensor(
                [forces['vdW'][(idx, )].epsilon._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    g.apply_nodes(
        lambda node: {'eq': torch.Tensor(
                [forces['vdW'][(idx, )].rmin_half._value for idx in range(g.number_of_nodes('atom'))])},
        ntype='atom')

    g = hgfp.mm.from_heterograph_with_xyz(g)

    g = hgfp.mm.energy_in_heterograph(g)

    u = torch.sum(
            torch.cat(
            [
                g.nodes['mol'].data['u' + term][:, None] for term in [
                    'bond', 'angle', 'torsion', 'one_four', 'nonbonded', '0'
            ]],
            dim=1),
        dim=1)

    return u
