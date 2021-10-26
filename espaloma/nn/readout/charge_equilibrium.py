""" Charge equilibrium.ß

"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_charges(node):
    """ Solve the function to get the absolute charges of atoms in a
    molecule from parameters.
    Parameters
    ----------
    e : tf.Tensor, dtype = tf.float32,
        electronegativity.
    s : tf.Tensor, dtype = tf.float32,
        hardness.
    Q : tf.Tensor, dtype = tf.float32, shape=(),
        total charge of a molecule.
    We use Lagrange multipliers to analytically give the solution.
    $$
    U({\bf q})
    &= \sum_{i=1}^N \left[ e_i q_i +  \frac{1}{2}  s_i q_i^2\right]
        - \lambda \, \left( \sum_{j=1}^N q_j - Q \right) \\
    &= \sum_{i=1}^N \left[
        (e_i - \lambda) q_i +  \frac{1}{2}  s_i q_i^2 \right
        ] + Q
    $$
    This gives us:
    $$
    q_i^*
    &= - e_i s_i^{-1}
    + \lambda s_i^{-1} \\
    &= - e_i s_i^{-1}
    + s_i^{-1} \frac{
        Q +
         \sum\limits_{i=1}^N e_i \, s_i^{-1}
        }{\sum\limits_{j=1}^N s_j^{-1}}
    $$
    """
    e = node.data["e"]
    s = node.data["s"]
    sum_e_s_inv = node.data["sum_e_s_inv"]
    sum_s_inv = node.data["sum_s_inv"]
    sum_q = node.data["sum_q"]

    return {
        "q_hat": -e * s ** -1
        + (s ** -1) * torch.div(sum_q + sum_e_s_inv, sum_s_inv)
    }


# =============================================================================
# MODULE CLASS
# =============================================================================
class ChargeEquilibrium(torch.nn.Module):
    """Charge equilibrium within batches of molecules."""

    def __init__(self):
        super(ChargeEquilibrium, self).__init__()

    def forward(self, g, total_charge=0.0):
        """ apply charge equilibrium to all molecules in batch """
        # calculate $s ^ {-1}$ and $ es ^ {-1}$
        import dgl
        g.apply_nodes(
            lambda node: {"s_inv": node.data["s"] ** -1}, ntype="n1"
        )

        g.apply_nodes(
            lambda node: {"e_s_inv": node.data["e"] * node.data["s"] ** -1},
            ntype="n1",
        )

        if "q" in g.nodes["n1"].data:
            # get total charge
            g.update_all(
                dgl.function.copy_src(src="q", out="m_q"),
                dgl.function.sum(msg="m_q", out="sum_q"),
                etype="n1_in_g",
            )
        else:
            g.nodes["g"].data["sum_q"] = (
                torch.ones(
                    g.batch_size,
                    1,
                    device=g.nodes["n1"].data["s"].device,
                )
                * total_charge
            )

        g.update_all(
            dgl.function.copy_src(src="sum_q", out="m_sum_q"),
            dgl.function.sum(msg="m_sum_q", out="sum_q"),
            etype="g_has_n1",
        )

        # get the sum of $s^{-1}$ and $m_s^{-1}$
        g.update_all(
            dgl.function.copy_src(src="s_inv", out="m_s_inv"),
            dgl.function.sum(msg="m_s_inv", out="sum_s_inv"),
            etype="n1_in_g",
        )

        g.update_all(
            dgl.function.copy_src(src="e_s_inv", out="m_e_s_inv"),
            dgl.function.sum(msg="m_e_s_inv", out="sum_e_s_inv"),
            etype="n1_in_g",
        )

        g.update_all(
            dgl.function.copy_src(src="sum_s_inv", out="m_sum_s_inv"),
            dgl.function.sum(msg="m_sum_s_inv", out="sum_s_inv"),
            etype="g_has_n1",
        )

        g.update_all(
            dgl.function.copy_src(src="sum_e_s_inv", out="m_sum_e_s_inv"),
            dgl.function.sum(msg="m_sum_e_s_inv", out="sum_e_s_inv"),
            etype="g_has_n1",
        )

        g.apply_nodes(get_charges, ntype="n1")

        return g
