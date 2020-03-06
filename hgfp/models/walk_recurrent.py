# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp

# =============================================================================
# MODULE CLASSES
# =============================================================================
class WRGN(torch.nn.Module):
    def __init__(self, units=32):
        super(WRGN, self).__init__()

        self.gru = torch.nn.GRU(
            units, units,
            num_layers=1,
            batch_first=True)

        self.d = torch.nn.Sequential(
            torch.nn.Linear(4 * units, units),
            torch.nn.Tanh(),
            torch.nn.Linear(units, units))

        self.units = units

    def gru_pool(self, g_idx):
        def _gru_pool(node):
            m = torch.stack(
                [node.data['h%s' % v_idx] for v_idx in range(g_idx)],
                dim=1)
            h, h_gru = self.gru(m)
            to_return =  {'h%s' % v_idx: h[:, v_idx, :] for v_idx in range(g_idx)}
            to_return.update(
                {'h_gru': h_gru[-1, :, :]})

            return to_return

        return _gru_pool

    def forward(self, g):

        g.multi_update_all(
            {
                'g1_as_%s_in_g%s' % (v_idx, g_idx):
                (
                    dgl.function.copy_src(src='h', out='m%s' % v_idx),
                    dgl.function.sum(msg='m%s' % v_idx, out='h%s' % v_idx)
                ) for v_idx in range(5) for g_idx in range(2, 5) if v_idx < g_idx
            },
            'sum')

        for g_idx in range(2, 5):
            g.apply_nodes(
                self.gru_pool(g_idx),
                ntype='g%s' % g_idx)

        g.multi_update_all(
            {
                'g%s_has_%s_g1' % (g_idx, v_idx):
                (
                    dgl.function.copy_src(src='h%s' % v_idx, out='m'),
                    dgl.function.sum(msg='m', out='h%s' % g_idx)
                ) for v_idx in range(5) for g_idx in range(2, 5) if v_idx < g_idx
            },
            'sum')

        # g.update_all(
        #     dgl.function.copy_src(src='h', out='m'),
        #     dgl.function.sum(msg='m', out='h'),
        #     etype='g1_in_ring')
        #
        # g.update_all(
        #     dgl.function.copy_src('h', out='m'),
        #     dgl.function.sum(msg='m', out='h_ring_down'),
        #     etype='ring_has_g1')

        g.apply_nodes(
            lambda node: {'h': self.d(
                torch.cat(
                    [node.data['h']] + [node.data['h%s' % g_idx] for g_idx in range(2, 5)],
                    dim=-1))},
            ntype='g1')

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
                    torch.nn.Linear(readout_units, 2)))


    def forward(self, g, return_graph=False):

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            lambda node: {'h': getattr(self, 'fr_atom')(node.data['h'])},
            etype='g1_in_atom')

        g.multi_update_all(
            {'g%s_in_%s' % (g_idx, term):
                (
                    dgl.function.copy_src(src='h_gru', out='m'),
                    dgl.function.sum(msg='m', out='h'),
                    lambda node: {'h': getattr(self, 'fr_%s' % term)(
                        node.data['h'])}
                ) for g_idx, term in {2: 'bond', 3: 'angle', 4: 'torsion'}.items()},
            'sum')

        for term in ['atom', 'bond', 'angle', 'torsion']:
            g.apply_nodes(
                lambda node: {
                    'k': node.data['h'][:, 0],
                    'eq': node.data['h'][:, 1]},
            ntype=term)

        return g

class Net(torch.nn.Module):
    def __init__(self, config, readout_units=128, input_units=128):
        super(Net, self).__init__()

        self.exes = []

        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(117, input_units),
            torch.nn.Tanh())

        def apply_atom_in_graph(fn):
            def _fn(g):
                g.apply_nodes(
                    lambda node: {'h': fn(node.data['h'])}, ntype='g1')
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
                    WRGN(exe))

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
                in_dim=input_units)

    def forward(self, g, return_graph=False):

        g.apply_nodes(
            lambda nodes: {'h': self.f_in(nodes.data['h0'])},
            ntype='g1')

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
