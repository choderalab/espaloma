""" Kipf and Welling GCN.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp

gcn_msg = dgl.function.copy_src(src='h', out='m')
gcn_reduce = dgl.function.sum(msg='m', out='h')

class NodeFullyConnect(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeFullyConnect, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h' : h}

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.apply_mod = NodeFullyConnect(in_dim, out_dim)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Net(torch.nn.Module):
    def __init__(self, config):
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

    def forward(self, g, training=True):
        x =  torch.zeros(
            g.ndata['atoms'].shape[0], 10, dtype=torch.float32)

        x[
            torch.arange(g.ndata['atoms'].shape[0]),
            torch.squeeze(g.ndata['atoms']).long()] = 1.0

        for exe in self.exes:
            if training == False:
                if exe.startswith('o'):
                    continue

            x = getattr(self, exe)(g, x)

        g.ndata['h'] = torch.squeeze(x)
        y = dgl.sum_nodes(g, 'h')
        return y
