# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import espaloma as esp


# =============================================================================
# CONSTANTS
# =============================================================================
from simtk import unit

# CODATA 2018
# ref https://en.wikipedia.org/wiki/Coulomb_constant
# K_E = (
#     8.9875517923 * 1e9
#     * unit.kilogram
#     * (unit.meter ** 3)
#     * (unit.second ** (-4))
#     * (unit.angstrom ** (-2))
#     ).value_in_unit(esp.units.COULOMB_CONSTANT_UNIT)


# =============================================================================
# UTILITY FUNCTIONS FOR COMBINATION RULES FOR NONBONDED
# =============================================================================
def geometric_mean(msg='m', out='k'):
    def _geometric_mean(nodes):
        return {out: torch.prod(nodes.mailbox[msg], dim=1).pow(0.5)}
    return _geometric_mean

def arithmetic_mean(msg='m', out='eq'):
    def _arithmetic_mean(nodes):
        return {out: torch.sum(nodes.mailbox[msg], dim=1).mul(0.5)}
    return _arithmetic_mean

# =============================================================================
# COMBINATION RULES FOR NONBONDED
# =============================================================================

def lorentz_berthelot(g):

    g.multi_update_all(
        {
            'n1_as_%s_in_%s' % (pos_idx, term): (
                dgl.function.copy_src(src='k', out='m_k'),
                geometric_mean(msg='m_k', out='k')
            ) for pos_idx in [0, 1] for term in ['nonbonded']
        },
        cross_reducer='sum'
    )

    g.multi_update_all(
        {
            'n1_as_%s_in_%s' % (pos_idx, term): (
                dgl.function.copy_src(src='eq', out='m_eq'),
                arithmetic_mean(msg='m_eq', out='eq')
            ) for pos_idx in [0, 1] for term in ['nonbonded']
        },
        cross_reducer='sum'
    )

    return g


# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================
def lj_12_6(x, k, eq):
    """ Lenard-Jones 12-6.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    k : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    eq : `torch.Tensor`,
        `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    """

    return esp.mm.functional.lj(x=x, k=k, eq=eq)

#
# def columb(x, q_prod, k_e=K_E):
#     """ Columb interaction without cutoff.
#
#     Parameters
#     ----------
#     x : `torch.Tensor`, shape=`(batch_size, 1)` or `(batch_size, batch_size, 1)`
#     q_prod : `torch.Tensor`,
#         `shape=(batch_size, 1) or `(batch_size, batch_size, 1)`
#
#     Returns
#     -------
#     u : `torch.Tensor`,
#         `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`
#
#
#     """
#     return k_e * x / q_prod
