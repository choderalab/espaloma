# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import hgfp

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HMP(torch.nn.Module):
    def __init__(self, units=32):
        super(HMP, self).__init__()
        for g_idx in range(2, 5):
            setattr(
                self,
                'd_up_%s' % g_idx,
                torch.nn.Sequential(
                    torch.nn.Linear(3 * units, units),
                    torch.nn.Tanh(),
                    torch.nn.Linear(units, units)))

        for g_idx in range(1, 4):
            setattr(
                self,
                'd_down_%s' % g_idx,
                torch.nn.Sequential(
                    torch.nn.Linear(2 * units, units),
                    torch.nn.Tanh(),
                    torch.nn.Linear(units, units)))

        self.gru = torch.nn.GRU(
            units, units,
            num_layers=1,
            batch_first=True)

        self.d_ring = torch.nn.Sequential(
            torch.nn.Linear(2 * units, units),
            torch.nn.Tanh(),
            torch.nn.Linear(units, units))

        self.units = units

    def gru_pool(self, node):
        m = node.mailbox['m']
        _, h = self.gru(m)
        h = h[-1, :, :]
        return {'h': h}

    def forward(self, g):
        g.nodes['g1'].data['h_ring_down'] = torch.zeros(
            (
                g.number_of_nodes(ntype='g1'),
                self.units
            ))

        # up
        for g_idx in range(2, 5):
            g.multi_update_all(
                {
                    'g%s_as_%s_in_g%s' % (g_idx-1, sub_g_idx, g_idx):(
                        dgl.function.copy_src(src='h', out='m'),
                        dgl.function.sum(msg='m', out='h%s' % sub_g_idx)
                    ) for sub_g_idx in range(2)
                },
                'sum')

            g.apply_nodes(
                lambda nodes: {'h': getattr(
                    self,
                    'd_up_%s' % g_idx)(torch.cat(
                    [nodes.data['h']] + [nodes.data['h%s'%sub_g_idx] for sub_g_idx in range(2)],
                    dim=-1))},
                ntype='g%s' % g_idx)

        # down
        for g_idx in range(4, 1, -1):
            g.multi_update_all(
                {
                    'g%s_has_%s_g%s' % (g_idx, sub_g_idx, g_idx-1):(
                        dgl.function.copy_src(src='h', out='m'),
                        dgl.function.sum(msg='m', out='h_down')) for sub_g_idx in range(2)
                },
                'sum')

            g.apply_nodes(
                lambda nodes: {'h': getattr(
                    self,
                    'd_down_%s' % (g_idx-1))(torch.cat(
                    [nodes.data['h'], nodes.data['h_down']], dim=-1))},
                ntype='g%s' % (g_idx - 1))

        # ring
        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='g1_in_ring')

        g.update_all(
            dgl.function.copy_src('h', out='m'),
            dgl.function.sum(msg='m', out='h_ring_down'),
            etype='ring_has_g1')

        g.apply_nodes(
            lambda nodes: {'h': self.d_ring(torch.cat(
                [nodes.data['h'], nodes.data['h_ring_down']], dim=-1))},
            ntype='g1')

        return g
