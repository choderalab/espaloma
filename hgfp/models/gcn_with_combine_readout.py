""" Kipf and Welling GCN, with readout function aggregating nodes into
hypernodes.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp

# =============================================================================
# MODULE CLASS
# =============================================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.apply_mod = hgfp.models.gcn.NodeFullyConnect(in_dim, out_dim)

    def forward(self, g, feature):
        # in this forward function we only operate on the nodes 'atom'
        g.nodes['atom'].data['h'] = feature

        g.multi_update_all(
            {'atom_neighbors_atom':(
                dgl.function.copy_src(src='h', out='m'),
                dgl.function.sum(msg='m', out='h'))},
            'sum')

        g.apply_nodes(func=self.apply_mod, ntype='atom')

        return g.nodes['atom'].data['h']

class ParamReadout(torch.nn.Module):
    def __init__(self, in_dim, readout_units=32):
        super(ParamReadout, self).__init__()

        for term in ['atom', 'bond', 'angle', 'torsion']:
            setattr(
                self,
                'fr_' + term,
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, readout_units),
                    torch.nn.Tanh(),
                    torch.nn.Linear(readout_units, 2),
                    ))

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

        k = k_and_eq[:, 0]
        eq = k_and_eq[:, 1]

        return {'k': k, 'eq': eq}

    def forward(self, g):

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='atom_in_mol')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='atom_in_bond')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_center'),
            etype='atom_as_center_in_angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_side'),
            etype='atom_as_side_in_angle')

        g.apply_nodes(
            lambda node: {'h': node.data['h_center'] + node.data['h_side']},
            ntype='angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='atom_as_side_in_angle')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_01'),
            etype='atom_as_0_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_01'),
            etype='atom_as_1_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_32'),
            etype='atom_as_2_in_torsion')

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h_01'),
            etype='atom_as_3_in_torsion')

        g.apply_nodes(
            lambda node: {'h': node.data['h_01'] + node.data['h_32']},
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
                lambda node: {'epsilon_pair': torch.sqrt(node.data['epsilon_pair'])},
                ntype=term)

        return g

class Net(torch.nn.Module):
    def __init__(self, config, readout_units=32):
        super(Net, self).__init__()

        dim = 10
        self.exes = []


        for idx, exe in enumerate(config):
            if exe.isnumeric():
                exe = float(exe)
                if exe >= 1:
                    exe = int(exe)

            if type(exe) == int:
                setattr(
                    self,
                    'd' + str(idx),
                    GCN(dim, exe))

                dim = exe
                self.exes.append('d' + str(idx))

            elif type(exe) == str:
                activation = getattr(torch.nn.functional, exe)

                setattr(
                    self,
                    'a' + str(idx),
                    lambda g, x: activation(x))
                self.exes.append('a' + str(idx))

            elif type(exe) == float:
                dropout = torch.nn.functional.Dropout
                setattr(
                    self,
                    'o' + str(idx),
                    lambda g, x: dropout(x, exe))
                self.exes.append('o' + str(idx))

            self.readout = ParamReadout(
                readout_units=readout_units,
                in_dim=dim)

    def forward(self, g, training=True):

        x =  torch.zeros(
            g.nodes['atom'].data['type'].shape[0], 10, dtype=torch.float32)

        x[
            torch.arange(g.nodes['atom'].data['type'].shape[0]),
            torch.squeeze(g.nodes['atom'].data['type']).long()] = 1.0

        for exe in self.exes:
            if training == False:
                if exe.startswith('o'):
                    continue

            x = getattr(self, exe)(g, x)

        g.nodes['atom'].data['h'] = x

        g = self.readout(g)

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
