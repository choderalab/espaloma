# =============================================================================
# IMPORTS
# =============================================================================
import math
import dgl
import torch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def reduce_stack(msg, out):
    """ Copy massage and stack. """

    def _reduce_stack(nodes, msg=msg, out=out):
        return {out: nodes.mailbox[msg]}

    return _reduce_stack


def copy_src(src, out):
    """ Copy source of an edge. """

    def _copy_src(edges, src=src, out=out):
        return {out: edges.src[src].clone()}

    return _copy_src


# =============================================================================
# SINGLE GEOMETRY ENTITY
# =============================================================================
def distance(x0, x1):
    """ Distance. """
    # TODO:
    # assertions / shape docstrings
    # p=2 indicates Euclidian distance
    return torch.norm(x0 - x1, p=2, dim=-1)


def _angle(r0, r1):
    """ Angle between vectors. """

    angle = torch.atan2(
        torch.norm(torch.cross(r0, r1), p=2, dim=-1),
        torch.sum(torch.mul(r0, r1), dim=-1),
    )

    return angle

def angle(x0, x1, x2):
    """ Angle between three points. """
    left = x1 - x0
    right = x1 - x2
    return _angle(left, right)


def _dihedral(r0, r1):
    """ Dihedral between normal vectors. """
    return _angle(r0, r1)

def dihedral(x0, x1, x2, x3, jitter=1e-10):
    """ Dihedral between four points. """
    # TODO:
    # check out
    # time-machine dihedral
    x0 = x0 + torch.randn_like(x0) * jitter
    x1 = x1 + torch.randn_like(x1) * jitter
    x2 = x2 + torch.randn_like(x2) * jitter
    x3 = x3 + torch.randn_like(x3) * jitter

    left = torch.cross(x1 - x0, x1 - x2)
    right = torch.cross(x2 - x1, x2 - x3)


    return _dihedral(left, right)

# =============================================================================
# GEOMETRY IN HYPERNODES
# =============================================================================
def apply_bond(nodes):
    """ Bond length in nodes. """
    return {
        "x": distance(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz1"]
        ),
    }

def apply_angle(nodes):
    """ Angle values in nodes. """
    return {
        "x": angle(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz1"],
            x2=nodes.data["xyz2"],
        ),
        "x_left": distance(
            x0=nodes.data["xyz1"],
            x1=nodes.data["xyz0"],
        ),
        "x_right": distance(
            x0=nodes.data["xyz1"],
            x1=nodes.data["xyz2"],
        ),
        "x_between": distance(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz2"],
        )
    }

def apply_torsion(nodes):
    """ Torsion dihedrals in nodes. """
    return {
        "x": dihedral(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz1"],
            x2=nodes.data["xyz2"],
            x3=nodes.data["xyz3"],
        ),
        "x_bond_left": distance(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz1"],
        ),
        "x_bond_center": distance(
            x0=nodes.data["xyz1"],
            x1=nodes.data["xyz2"],
        ),
        "x_bond_right": distance(
            x0=nodes.data["xyz2"],
            x1=nodes.data["xyz3"],
        ),
        "x_angle_left": angle(
            x0=nodes.data["xyz0"],
            x1=nodes.data["xyz1"],
            x2=nodes.data["xyz2"],
        ),
        "x_angle_right": angle(
            x0=nodes.data["xyz1"],
            x1=nodes.data["xyz2"],
            x2=nodes.data["xyz3"],
        )
    }

# =============================================================================
# GEOMETRY IN GRAPH
# =============================================================================
# NOTE:
# The following functions modify graphs in-place.


def geometry_in_graph(g):
    """ Assign values to geometric entities in graphs.

    Parameters
    ----------
    g : `dgl.DGLHeteroGraph`
        Input graph.

    Returns
    -------
    g : `dgl.DGLHeteroGraph`
        Output graph.

    Notes
    -----
    This function modifies graphs in-place.

    """

    # Copy coordinates to higher-order nodes.
    g.multi_update_all(
        {
            **{
                "n1_as_%s_in_n%s"
                % (pos_idx, big_idx): (
                    copy_src(src="xyz", out="m_xyz%s" % pos_idx),
                    dgl.function.sum(
                        msg="m_xyz%s" % pos_idx, out="xyz%s" % pos_idx
                    ),
                )
                for big_idx in range(2, 5)
                for pos_idx in range(big_idx)
            },
            **{
                "n1_as_%s_in_%s"
                % (pos_idx, term): (
                    copy_src(src="xyz", out="m_xyz%s" % pos_idx),
                    dgl.function.sum(
                        msg="m_xyz%s" % pos_idx, out="xyz%s" % pos_idx
                    ),
                )
                for term in ["nonbonded", "onefour"]
                for pos_idx in [0, 1]
            },
        },
        cross_reducer="sum",
    )

    # apply geometry functions
    g.apply_nodes(apply_bond, ntype="n2")
    g.apply_nodes(apply_angle, ntype="n3")

    if g.number_of_nodes("n4") > 0:
        g.apply_nodes(apply_torsion, ntype="n4")

    # copy coordinates to nonbonded
    if g.number_of_nodes("nonbonded") > 0:
        g.apply_nodes(apply_bond, ntype="nonbonded")

    if g.number_of_nodes("onefour") > 0:
        g.apply_nodes(apply_bond, ntype="onefour")

    return g


class GeometryInGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GeometryInGraph, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return geometry_in_graph(g, *self.args, **self.kwargs)
