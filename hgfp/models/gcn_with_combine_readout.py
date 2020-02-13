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
        self.apply_mod = hgfp.gcn.NodeFullyConnect(in_dim, out_dim)

    def forward(self, g, feature):
        # in this forward function we only operate on the nodes 'atom'
        g.nodes['atom'].data['h'] = feature

        g.multi_update_all(
            {'atom_neighbors_atom':(
                dgl.function.copy_src(src='h', out='m'),
                dgl.function.sum(msg='m', out='h'))},
            'sum')

        g.apply_nodes(func=self.apply_mod, ntype='atom')

        return g.node['atom'].data['h']

class ParamReadout(torch.nn.Module):
    def __init__(self, config, in_dim, readout_units=32):
        super(ParamReadout, self).__init__()
        # atom readout
        self.fr_atom = torch.nn.Sequential(
            torch.nn.Linear(in_dim, readout_units),
            torch.nn.functional.tanh(),
            torch.nn.Linear(readout_units, 2))

        # bond readout
        # outputs k, eq
        self.fr_bond = torch.nn.Sequential(
            torch.nn.Linear(in_dim, readout_units),
            torch.nn.functional.tanh(),
            torch.nn.Linear(readout_units, 2))

        # angle readout
        # outputs k, eq
        self.fr_angle = torch.nn.Sequential(
            torch.nn.Linear(in_dim, readout_units),
            torch.nn.functional.tanh(),
            torch.nn.Linear(readout_units, 2))

        # torsion readout
        # outputs k, eq
        self.fr_torsion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, readout_units),
            torch.nn.functional.tanh(),
            torch.nn.Linear(readout_units, 2))

    def apply_node(self, node, fn):
        h = node.data['h']
        k_and_eq = torch.squeeze(fn(h))
        k = k_and_eq[0]
        eq = k_and_eq[1]

        return {'k': k, 'eq': eq}

    def forward(self, g):
        g.multi_update_all(
            {
                'atom_in_bond':(
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h')),
                'atom_as_center_in_angle':(
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h'))),
                'atom_as_side_in_angle':(
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h'))),
                'atom_as_center_in_torsion':(
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h'))),
                'atom_as_side_in_torsion':(
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h')))
            },
            'stack')
        return g

class Net(hgfp.models.gcn.Net):
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


    def foward(self, g, training=True):
        x =  torch.zeros(
            nodes['atom'].data['type'].shape[0], 10, dtype=torch.float32)

        x[
            torch.arange(nodes['atom'].data['type'].shape[0]),
            torch.squeeze(nodes['atom'].data['type']).long()] = 1.0

        for exe in self.exes:
            if training == False:
                if exe.startswith('o'):
                    continue

            x = getattr(self, exe)(g, x)

        g.nodes.data['h'] = torch.squeeze(x)
        y = dgl.sum_nodes(g, 'h')
        return y
