# =============================================================================
# IMPORTS
# =============================================================================
import torch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def reduce_stack(msg, out):
    """Copy massage and stack."""

    def _reduce_stack(nodes, msg=msg, out=out):
        return {out: nodes.mailbox[msg]}

    return _reduce_stack


def copy_src(src, out):
    """Copy source of an edge."""

    def _copy_src(edges, src=src, out=out):
        return {out: edges.src[src].clone()}

    return _copy_src


# =============================================================================
# SINGLE GEOMETRY ENTITY
# =============================================================================
def distance(x0, x1):
    """Distance."""
    return torch.norm(x0 - x1, p=2, dim=-1)


def _angle(r0, r1):
    """Angle between vectors."""

    angle = torch.atan2(
        torch.norm(torch.cross(r0, r1), p=2, dim=-1),
        torch.sum(torch.mul(r0, r1), dim=-1),
    )

    return angle


def angle(x0, x1, x2):
    """Angle between three points."""
    left = x1 - x0
    right = x1 - x2
    return _angle(left, right)


def _dihedral(r0, r1):
    """Dihedral between normal vectors."""
    return _angle(r0, r1)


# TODO check
def dihedral(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
) -> torch.Tensor:
    """Dihedral between four points.

    Reference
    ---------
    Closely follows implementation in Yutong Zhao's timemachine:
        https://github.com/proteneer/timemachine/blob/1a0ab45e605dc1e28c44ea90f38cb0dedce5c4db/timemachine/potentials/bonded.py#L152-L199
    """
    # check input shapes

    assert x0.shape == x1.shape == x2.shape == x3.shape

    # compute displacements 0->1, 2->1, 2->3
    r01 = x1 - x0 + torch.randn_like(x0) * 1e-5
    r21 = x1 - x2 + torch.randn_like(x0) * 1e-5
    r23 = x3 - x2 + torch.randn_like(x0) * 1e-5

    # compute normal planes
    n1 = torch.cross(r01, r21)
    n2 = torch.cross(r21, r23)

    rkj_normed = r21 / torch.norm(r21, dim=-1, keepdim=True)

    y = torch.sum(torch.mul(torch.cross(n1, n2), rkj_normed), dim=-1)
    x = torch.sum(torch.mul(n1, n2), dim=-1)

    # choose quadrant correctly
    theta = torch.atan2(y, x)

    return theta

# TODO(gianscarpe) check implementation (ask for clarification)
def oop(
    i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, l: torch.Tensor
) -> torch.Tensor:
    """

    Ref http://www.ccl.net/chemistry/resources/messages/1996/09/19.008-dir/

    l ikj -> 1 234
    """
    # compute displacements 0->1, 2->1, 2->3
    e_ji = j - i + torch.randn_like(i) * 1e-11
    e_jk = j - k + torch.randn_like(i) * 1e-11
    e_jl = j - l + torch.randn_like(i) * 1e-11

    ejl_normed = e_jl / torch.norm(e_jl, dim=-1, keepdim=True)
    eji_normed = e_ji / torch.norm(e_ji, dim=-1, keepdim=True)
    ejk_normed = e_jk / torch.norm(e_jk, dim=-1, keepdim=True)

    phi = angle(i, j, k)
    phi = phi + torch.randn_like(phi) * 1e-11
    n_residues = i.shape[1]


    out = torch.arcsin(((torch.cross(eji_normed, ejk_normed)) / torch.sin(phi)[:, :, None] @ ejl_normed.permute(0, 2, 1))[:, torch.arange(n_residues), torch.arange(n_residues)])
    
    return out


# =============================================================================
# GEOMETRY IN HYPERNODES
# =============================================================================
def apply_bond(nodes):
    """Bond length in nodes."""

    return {"x": distance(x0=nodes.data["xyz0"], x1=nodes.data["xyz1"])}


def apply_angle(nodes):
    """Angle values in nodes."""
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
        ),
    }


def apply_torsion(nodes):
    """Torsion dihedrals in nodes."""
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
        ),
    }

def apply_oop(nodes):
    """Torsion dihedrals in nodes."""
    return {
        "x": oop(
            j=nodes.data["xyz0"],
            i=nodes.data["xyz1"],
            k=nodes.data["xyz2"],
            l=nodes.data["xyz3"],
        )}

# =============================================================================
# GEOMETRY IN GRAPH
# =============================================================================
# NOTE:
# The following functions modify graphs in-place.


def geometry_in_graph(g):
    """Assign values to geometric entities in graphs.

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
    import dgl

    # Copy coordinates to higher-order nodes.
    
    g.multi_update_all(
        {
            **{
                "n1_as_%s_in_n%s"
                % (pos_idx, big_idx): (
                    dgl.function.copy_u(u="xyz", out="m_xyz%s" % pos_idx),
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
                    dgl.function.copy_u(u="xyz", out="m_xyz%s" % pos_idx),
                    dgl.function.sum(
                        msg="m_xyz%s" % pos_idx, out="xyz%s" % pos_idx
                    ),
                )
                for term in ["nonbonded", "onefour"]
                for pos_idx in [0, 1]
            },
            **{
                "n1_as_%s_in_%s"
                % (pos_idx, term): (
                    dgl.function.copy_u(u="xyz", out="m_xyz%s" % pos_idx),
                    dgl.function.sum(
                        msg="m_xyz%s" % pos_idx, out="xyz%s" % pos_idx
                    ),
                )
                for term in ["n4_improper"]
                for pos_idx in [0, 1, 2, 3]
            },
            **{
                "n1_as_%s_in_%s"
                % (pos_idx, term): (
                    dgl.function.copy_u(u="xyz", out="m_xyz%s" % pos_idx),
                    dgl.function.sum(
                        msg="m_xyz%s" % pos_idx, out="xyz%s" % pos_idx
                    ),
                )
                for term in ["n4_oop"]
                for pos_idx in [0, 1, 2, 3]
                if "n4_oop" in g._ntypes
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

        
    if g.number_of_nodes("n4_improper") > 0:
        g.apply_nodes(apply_torsion, ntype="n4_improper")


    if "n4_oop" in g._ntypes and g.number_of_nodes("n4_oop") > 0:

        g.apply_nodes(apply_oop, ntype="n4_oop")

    return g


class GeometryInGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(GeometryInGraph, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return geometry_in_graph(g, *self.args, **self.kwargs)
