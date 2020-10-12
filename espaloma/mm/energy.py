# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch

import espaloma as esp


# =============================================================================
# ENERGY IN HYPERNODES---BONDED
# =============================================================================
def apply_bond(nodes, suffix=""):
    """ Bond energy in nodes. """
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.harmonic_bond(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
            eq=nodes.data["eq%s" % suffix],
        )
    }

    # else:
    #     return {
    #         'u%s' % suffix: esp.mm.bond.harmonic_bond_re(
    #             x=nodes.data['x'],
    #             k=nodes.data['k%s' % suffix],
    #             eq=nodes.data['eq%s' % suffix],
    #         )
    #     }

def apply_angle(nodes, suffix=""):
    """ Angle energy in nodes. """
    return {
        "u%s"
        % suffix: esp.mm.angle.harmonic_angle(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
            eq=nodes.data["eq%s" % suffix],
        )
    }


def apply_torsion(nodes, suffix=""):
    """ Torsion energy in nodes. """
    if "phases%s" % suffix in nodes.data and "periodicity%s" % suffix in nodes.data:
        return {
            "u%s"
            % suffix: esp.mm.torsion.periodic_torsion(
                x=nodes.data["x"],
                k=nodes.data["k%s" % suffix],
                phases=nodes.data["phases%s" % suffix],
                periodicity=nodes.data["periodicity%s" % suffix],
            )
        }


    else:
        return {
            "u%s"
            % suffix: esp.mm.torsion.periodic_torsion(
                x=nodes.data["x"],
                k=nodes.data["k%s" % suffix],
            )
        }


def apply_improper_torsion(nodes, suffix=""):
    # TODO: decide if "trefoil" convention is better handled here or in pooling
    raise (NotImplementedError)


def apply_bond_gaussian(nodes, suffix=""):
    """ Bond energy in nodes. """
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.gaussian_bond(
            x=nodes.data["x"],
            coefficients=nodes.data["coefficients%s" % suffix],
        )
    }

def apply_bond_linear_mixture(nodes, suffix=""):
    """ Bond energy in nodes. """
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.linear_mixture_bond(
            x=nodes.data["x"],
            coefficients=nodes.data["coefficients%s" % suffix],
        )
    }

# =============================================================================
# ENERGY IN HYPERNODES---NONBONDED
# =============================================================================
def apply_nonbonded(nodes, scaling=1.0, suffix=""):
    """ Nonbonded in nodes. """
    # TODO: should this be 9-6 or 12-6?
    return {
        "u%s"
        % suffix: scaling * esp.mm.nonbonded.lj_9_6(
            x=nodes.data["x"],
            sigma=nodes.data["sigma%s" % suffix],
            epsilon=nodes.data["epsilon%s" % suffix],
        )
    }


# =============================================================================
# ENERGY IN GRAPH
# =============================================================================
def energy_in_graph(g, suffix="", terms=["n2", "n3", "n4"]): # "onefour", "nonbonded"]):
    """ Calculate the energy of a small molecule given parameters and geometry.

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
    # TODO: this is all very restricted for now
    # we need to make this better

    if "nonbonded" in terms or "onefour" in terms:
        # apply combination rule
        esp.mm.nonbonded.lorentz_berthelot(g, suffix=suffix)

    if "n2" in terms:
        # apply energy function
        g.apply_nodes(
            lambda node: apply_bond(node, suffix=suffix),
            ntype="n2",
        )


    if "n3" in terms:
        g.apply_nodes(
            lambda node: apply_angle(node, suffix=suffix), ntype="n3",
        )

    if g.number_of_nodes("n4") > 0 and "n4" in terms:
        g.apply_nodes(
            lambda node: apply_torsion(node, suffix=suffix),
            ntype="n4",
        )
    if g.number_of_nodes("n4_improper") > 0 and "n4_improper" in terms:
        g.apply_nodes(
            lambda node: apply_improper_torsion(node, suffix=suffix),
            ntype="n4_improper",
        )

    if g.number_of_nodes("nonbonded") > 0 and "nonbonded" in terms:
        g.apply_nodes(
            lambda node: apply_nonbonded(node, suffix=suffix),
            ntype="nonbonded",
        )

    if g.number_of_nodes("onefour") > 0 and "onefour" in terms:
        g.apply_nodes(
            lambda node: apply_nonbonded(
                node, suffix=suffix, scaling=0.5,
            ), ntype="onefour"
        )

    # sum up energy
    # bonded
    g.multi_update_all(
        {
            "%s_in_g"
            % term: (
                dgl.function.copy_src(src="u%s" % suffix, out="m_%s" % term),
                dgl.function.sum(
                    msg="m_%s" % term, out="u_%s%s" % (term, suffix)
                ),
            )
            for term in terms
        },
        cross_reducer="sum",
    )

    g.apply_nodes(
        lambda node: {
            "u%s"
            % suffix: sum(
                node.data["u_%s%s" % (term, suffix)] for term in terms
            )
        },
        ntype="g",
    )

    if 'u0' in g.nodes['g'].data:
        g.apply_nodes(
            lambda node: {'u': node.data['u'] + node.data['u0']},
            ntype="g",
        )

    return g


class EnergyInGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(EnergyInGraph, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return energy_in_graph(g, *self.args, **self.kwargs)
