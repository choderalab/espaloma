""" Kipf and Welling GCN, with readout function aggregating nodes into
hypernodes.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp
import math
from dgl.nn import pytorch as dgl_pytorch

# =============================================================================
# MODULE CLASS
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
            self,
            in_feat,
            out_feat,
            model=dgl_pytorch.conv.SAGEConv,
            kwargs={'aggregator_type': 'mean'}):
        super(GN, self).__init__()
        self.gn = model(in_feat, out_feat, **kwargs)

    def forward(self, g):
        x = g.nodes['atom'].data['h']
        g_sub = dgl.to_homo(
            g.edge_type_subgraph(['atom_neighbors_atom']))
        x = self.gn(g_sub, x)
        g.nodes['atom'].data['h'] = x
        return g

class ParamReadout(torch.nn.Module):
    def __init__(self, in_dim, readout_units=128):
        super(ParamReadout, self).__init__()

        for term in ['atom', 'bond', 'angle', 'torsion']:
            setattr(
                self,
                'fr_' + term,
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, readout_units),
                    torch.nn.Linear(readout_units, 2),
                    ))

        self.fr_angle_0 = torch.nn.Linear(3 * in_dim, in_dim)
        self.fr_torsion_0 = torch.nn.Linear(4 * in_dim, in_dim)
        self.fr_bond_0 = torch.nn.Linear(2 * in_dim, in_dim)

        setattr(
            self,
            'fr_mol',
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, readout_units),
                torch.nn.Tanh(),
                torch.nn.Linear(readout_units, 1)))

    def apply_node(self, node, fn):
        h = node.data['h']

        # everything should be positive
        k_and_eq = torch.abs(fn(h))
        # k_and_eq = fn(h)
        k = k_and_eq[:, 0]
        eq = k_and_eq[:, 1]

        return {'k': k, 'eq': eq}

    def forward(self, g):

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='atom_in_mol')

        g.update_all(
            dgl.function.copy_src(src='h', out='m0'),
            dgl.function.sum(msg='m0', out='h0'),
            etype='atom_as_0_in_bond')

        g.update_all(
            dgl.function.copy_src(src='h', out='m1'),
            dgl.function.sum(msg='m1', out='h1'),
            etype='atom_as_1_in_bond')

        g.apply_nodes(
            lambda node: {'h01': torch.cat(
                [
                    node.data['h0'],
                    node.data['h1'],
                ],
                axis=-1
            )},
            ntype='bond')

        g.apply_nodes(
            lambda node: {'h10': torch.cat(
                [
                    node.data['h1'],
                    node.data['h0'],
                ],
                axis=-1
            )},
            ntype='bond')

        g.apply_nodes(
            lambda node: {'h':
                self.fr_bond_0(node.data['h01']) + self.fr_bond_0(node.data['h10'])},
            ntype='bond')

        g.update_all(
            dgl.function.copy_src(src='h', out='m0'),
            dgl.function.sum(msg='m0', out='h0'),
            etype='atom_as_0_in_angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m1'),
            dgl.function.sum(msg='m1', out='h1'),
            etype='atom_as_1_in_angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m2'),
            dgl.function.sum(msg='m2', out='h2'),
            etype='atom_as_2_in_angle')

        g.apply_nodes(
            lambda node: {'h012': torch.cat(
                [
                    node.data['h0'],
                    node.data['h1'],
                    node.data['h2']
                ],
                axis=-1
            )},
            ntype='angle')

        g.apply_nodes(
            lambda node: {'h210': torch.cat(
                [
                    node.data['h2'],
                    node.data['h1'],
                    node.data['h0']
                ],
                axis=-1
            )},
            ntype='angle')

        g.apply_nodes(
            lambda node: {'h':
                self.fr_angle_0(node.data['h012']) + self.fr_angle_0(node.data['h210'])},
            ntype='angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m0'),
            dgl.function.sum(msg='m0', out='h0'),
            etype='atom_as_0_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m1'),
            dgl.function.sum(msg='m1', out='h1'),
            etype='atom_as_1_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m2'),
            dgl.function.sum(msg='m2', out='h2'),
            etype='atom_as_2_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m3'),
            dgl.function.sum(msg='m3', out='h3'),
            etype='atom_as_3_in_torsion')

        g.apply_nodes(
            lambda node: {'h0123': torch.cat(
                [
                    node.data['h0'],
                    node.data['h1'],
                    node.data['h2'],
                    node.data['h3']
                ],
                axis=-1
            )},
            ntype='torsion')

        g.apply_nodes(
            lambda node: {'h3210': torch.cat(
                [
                    node.data['h3'],
                    node.data['h2'],
                    node.data['h1'],
                    node.data['h0']
                ],
                axis=-1
            )},
            ntype='torsion')

        g.apply_nodes(
            lambda node: {'h':
                self.fr_torsion_0(node.data['h0123']) + self.fr_torsion_0(node.data['h3210'])},
            ntype='torsion')

        for term in ['atom', 'bond', 'angle', 'torsion']:
            g.apply_nodes(
                lambda node: self.apply_node(node, fn=getattr(self, 'fr_' + term)),
                ntype=term)

        g.apply_nodes(
            lambda node: {'u0': torch.squeeze(self.fr_mol(node.data['h']))},
            ntype='mol')

        # combine sigma and epsilon

        g.multi_update_all(
            {
                'atom_in_one_four':(
                    dgl.function.copy_src(src='k', out='epsilon'),
                    dgl.function.prod(msg='epsilon', out='epsilon_pair')),
                'atom_in_nonbonded':(
                    dgl.function.copy_src(src='k', out='epsilon'),
                    dgl.function.prod(msg='epsilon', out='epsilon_pair'))
            },
            'stack')


        g.multi_update_all(
            {
                'atom_in_one_four':(
                    dgl.function.copy_src(src='eq', out='sigma'),
                    dgl.function.prod(msg='sigma', out='sigma_pair')),
                'atom_in_nonbonded':(
                    dgl.function.copy_src(src='eq', out='sigma'),
                    dgl.function.prod(msg='sigma', out='sigma_pair'))
            },
            'stack')


        for term in ['one_four', 'nonbonded']:
            g.apply_nodes(
                lambda node: {'epsilon_pair': torch.sqrt(torch.abs(node.data['epsilon_pair']))},
                ntype=term)

        return g

class Net(torch.nn.Module):
    def __init__(self, config, readout_units=128, input_units=128):
        super(Net, self).__init__()

        dim = input_units
        self.exes = []

        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(117, input_units),
            torch.nn.Tanh())

        def apply_atom_in_graph(fn):
            def _fn(g):
                g.apply_nodes(
                    lambda node: {'h': fn(node.data['h'])}, ntype='atom')
                return g
            return _fn

        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except:
                pass

            if type(exe) == int:
                setattr(
                    self,
                    'd' + str(idx),
                    GN(dim, exe))

                dim = exe
                self.exes.append('d' + str(idx))

            elif type(exe) == str:
                activation = getattr(torch.nn.functional, exe)

                setattr(
                    self,
                    'a' + str(idx),
                    apply_atom_in_graph(activation))

                self.exes.append('a' + str(idx))

            elif type(exe) == float:
                dropout = torch.nn.Dropout(exe)
                setattr(
                    self,
                    'o' + str(idx),
                    apply_atom_in_graph(dropout))

                self.exes.append('o' + str(idx))

            self.readout = ParamReadout(
                readout_units=readout_units,
                in_dim=dim)

    def forward(self, g, return_graph=False):

        g.apply_nodes(
            lambda nodes: {'h': self.f_in(nodes.data['h0'])},
            ntype='atom')

        for exe in self.exes:
            g = getattr(self, exe)(g)

        g = self.readout(g)

        if return_graph == True:
            return g

        g = hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(
            g)

        g = hgfp.mm.energy_in_heterograph.u(g)

        u = torch.sum(
                torch.cat(
                [
                    g.nodes['mol'].data['u' + term][:, None] for term in [
                        'bond', 'angle', 'torsion', 'one_four', 'nonbonded', '0'
                ]],
                dim=1),
            dim=1)


        return u
