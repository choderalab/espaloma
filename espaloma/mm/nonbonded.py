# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# CONSTANTS
# =============================================================================
import espaloma as esp
from openmm import unit

# CODATA 2018
# ref https://en.wikipedia.org/wiki/Coulomb_constant
# Coulomb constant
K_E = (
    8.9875517923
    * 1e9
    * unit.newton
    * unit.meter**2
    * unit.coulomb ** (-2)
    * esp.units.PARTICLE ** (-1)
).value_in_unit(esp.units.COULOMB_CONSTANT_UNIT)

# =============================================================================
# UTILITY FUNCTIONS FOR COMBINATION RULES FOR NONBONDED
# =============================================================================
def geometric_mean(msg="m", out="epsilon"):
    def _geometric_mean(nodes):
        return {out: torch.prod(nodes.mailbox[msg], dim=1).pow(0.5)}

    return _geometric_mean


def arithmetic_mean(msg="m", out="sigma"):
    def _arithmetic_mean(nodes):
        return {out: torch.sum(nodes.mailbox[msg], dim=1).mul(0.5)}

    return _arithmetic_mean


# =============================================================================
# COMBINATION RULES FOR NONBONDED
# =============================================================================
def lorentz_berthelot(g, suffix=""):
    import dgl

    g.multi_update_all(
        {
            "n1_as_%s_in_%s"
            % (pos_idx, term): (
                dgl.function.copy_u(
                    u="epsilon%s" % suffix, out="m_epsilon"
                ),
                geometric_mean(msg="m_epsilon", out="epsilon%s" % suffix),
            )
            for pos_idx in [0, 1]
            for term in ["nonbonded", "onefour"]
        },
        cross_reducer="sum",
    )

    g.multi_update_all(
        {
            "n1_as_%s_in_%s"
            % (pos_idx, term): (
                dgl.function.copy_u(u="sigma%s" % suffix, out="m_sigma"),
                arithmetic_mean(msg="m_sigma", out="sigma%s" % suffix),
            )
            for pos_idx in [0, 1]
            for term in ["nonbonded", "onefour"]
        },
        cross_reducer="sum",
    )

    return g


def multiply_charges(g, suffix=""):
    """Multiply the charges of atoms into nonbonded and onefour terms.

    Parameters
    ----------
    g : dgl.HeteroGraph
        Input graph.

    Returns
    -------
    dgl.HeteroGraph : The modified graph with charges.

    """
    import dgl

    g.multi_update_all(
        {
            "n1_as_%s_in_%s"
            % (pos_idx, term): (
                dgl.function.copy_u(u="q%s" % suffix, out="m_q"),
                dgl.function.sum(msg="m_q", out="_q")
                # lambda node: {"q%s" % suffix: node.mailbox["m_q"].prod(dim=1)}
            )
            for pos_idx in [0, 1]
            for term in ["nonbonded", "onefour"]
        },
        cross_reducer="stack",
        apply_node_func=lambda node: {"q": node.data["_q"].prod(dim=1)},
    )

    return g


# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================
def lj_12_6(x, sigma, epsilon):
    """Lennard-Jones 12-6.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    sigma : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    epsilon : `torch.Tensor`,
        `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    """

    return esp.mm.functional.lj(x=x, sigma=sigma, epsilon=epsilon)


def lj_9_6(x, sigma, epsilon):
    """Lennard-Jones 9-6.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    sigma : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    epsilon : `torch.Tensor`,
        `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    Returns
    -------
    u : `torch.Tensor`, `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`

    """

    return esp.mm.functional.lj(
        x=x, sigma=sigma, epsilon=epsilon, order=[9, 6], coefficients=[2, 3]
    )


def coulomb(x, q, k_e=K_E):
    """Columb interaction without cutoff.

    Parameters
    ----------
    x : `torch.Tensor`, shape=`(batch_size, 1)` or `(batch_size, batch_size, 1)`
        Distance between atoms.

    q : `torch.Tensor`,
        `shape=(batch_size, 1) or `(batch_size, batch_size, 1)`
        Product of charge.

    Returns
    -------
    torch.Tensor : `shape=(batch_size, 1)` or `(batch_size, batch_size, 1)`
        Coulomb energy.

    Notes
    -----
    This computes half Coulomb energy to count for the duplication in onefour
        and nonbonded enumerations.

    """
    return 0.5 * k_e * q / x
