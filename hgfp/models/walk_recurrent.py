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
            torch.nn.Linear(5 * units, units),
            torch.nn.Tanh(),
            torch.nn.Linear(units, units))

        self.units = units

    def gru_pool(self, g_idx):
        def _gru_pool(node):
            m = torch.stack(
                [node.data['h%s' % v_idx] for v_idx in range(g_idx)],
                dim=1)
            h, _ = self.gru(m)
            return {'h%s' % v_idx: h[:, v_idx, :] for v_idx in range(g_idx)}
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

        g.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h'),
            etype='g1_in_ring')

        g.update_all(
            dgl.function.copy_src('h', out='m'),
            dgl.function.sum(msg='m', out='h_ring_down'),
            etype='ring_has_g1')

        g.apply_nodes(
            lambda node: {'h': self.d(
                torch.cat(
                    [node.data['h']] + [node.data['h_ring_down']] + [node.data['h%s' % g_idx] for g_idx in range(2, 5)],
                    dim=-1))},
            ntype='g1')



        return g
