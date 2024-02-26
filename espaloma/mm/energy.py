# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import torch


# =============================================================================
# ENERGY IN HYPERNODES---BONDED
# =============================================================================
def apply_bond(nodes, suffix=""):
    """Bond energy in nodes."""
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.harmonic_bond(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
            eq=nodes.data["eq%s" % suffix],
        )
    }


def apply_bond_mmff(nodes, suffix=""):
    """Bond energy in nodes."""
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.harmonic_bond_mmff( # change for MMFF
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

def apply_stretch_bend(nodes, suffix):
    """
    TODO copy from contribution 2 and 3 into eq 5
    """
    
    return {
        "u%s"
        % suffix: esp.mm.angle.harmonic_stretch_bend_mmff( # change for MMFF
            x=nodes.data["x"],
            eq=nodes.data['eq'],
            k=nodes.data["kstretch"],
            # Coming from n2
            eq_ij=nodes.data['eq2_ij'],
            eq_kj=nodes.data['eq2_kj'],
            x_ij=nodes.data['x2_ij'],
            x_kj=nodes.data['x2_kj'],
            is_linear=nodes.data['lin']
        )
    }

def apply_angle(nodes, suffix=""):
    """Angle energy in nodes."""
    return {
        "u%s"
        % suffix: esp.mm.angle.harmonic_angle(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
            eq=nodes.data["eq%s" % suffix],
        )
    }

def apply_angle_mmff(nodes, suffix=""):

    """Angle energy in nodes."""
    return {
        "u%s"
        % suffix: esp.mm.angle.harmonic_angle_mmff(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
            eq=nodes.data["eq%s" % suffix],
            lin=nodes.data["lin"]
        )
    }




def apply_angle_ii(nodes, suffix=""):
    return {
        # "u_angle_high%s"
        # % suffix: esp.mm.angle.angle_high(
        #     u_angle=nodes.data["u"],
        #     k3=nodes.data["k3"],
        #     k4=nodes.data["k4"],
        # ),
        "u_urey_bradley%s"
        % suffix: esp.mm.angle.urey_bradley(
            x_between=nodes.data["x_between"],
            coefficients=nodes.data["coefficients_urey_bradley"],
            phases=[0.0, 12.0],
        ),
        "u_bond_bond%s"
        % suffix: esp.mm.angle.bond_bond(
            u_left=nodes.data["u_left"],
            u_right=nodes.data["u_right"],
            k_bond_bond=nodes.data["k_bond_bond"],
        ),
        "u_bond_angle%s"
        % suffix: esp.mm.angle.bond_angle(
            u_left=nodes.data["u_left"],
            u_right=nodes.data["u_right"],
            u_angle=nodes.data["u"],
            k_bond_angle=nodes.data["k_bond_angle"],
        ),
    }


def apply_bond_ii(nodes, suffix=""):
    return {
        "u_bond_high%s"
        % suffix: esp.mm.bond.bond_high(
            u_bond=nodes.data["u"],
            k3=nodes.data["k3"],
            k4=nodes.data["k4"],
        )
    }


def apply_torsion_ii(nodes, suffix=""):
    """Torsion energy in nodes."""
    return {
        "u_angle_angle%s"
        % suffix: esp.mm.torsion.angle_angle(
            u_angle_left=nodes.data["u_angle_left"],
            u_angle_right=nodes.data["u_angle_right"],
            k_angle_angle=nodes.data["k_angle_angle"],
        ),
        "u_angle_torsion%s"
        % suffix: esp.mm.torsion.angle_torsion(
            u_angle_left=nodes.data["u_angle_left"],
            u_angle_right=nodes.data["u_angle_right"],
            u_torsion=nodes.data["u"],
            k_angle_torsion=nodes.data["k_angle_torsion"],
        ),
        "u_angle_angle_torsion%s"
        % suffix: esp.mm.torsion.angle_angle_torsion(
            u_angle_left=nodes.data["u_angle_left"],
            u_angle_right=nodes.data["u_angle_right"],
            u_torsion=nodes.data["u"],
            k_angle_angle_torsion=nodes.data["k_angle_angle_torsion"],
        ),
        "u_bond_torsion%s"
        % suffix: esp.mm.torsion.bond_torsion(
            u_bond_left=nodes.data["u_bond_left"],
            u_bond_right=nodes.data["u_bond_right"],
            u_bond_center=nodes.data["u_bond_center"],
            u_torsion=nodes.data["u"],
            k_side_torsion=nodes.data["k_side_torsion"],
            k_center_torsion=nodes.data["k_center_torsion"],
        ),
    }


def apply_torsion(nodes, suffix=""):
    """Torsion energy in nodes."""
    if (
        "phases%s" % suffix in nodes.data
        and "periodicity%s" % suffix in nodes.data
    ):
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


def apply_torsion(nodes, suffix=""):
    """Torsion energy in nodes."""
    return {
        "u%s"
        % suffix: esp.mm.torsion.periodic_torsion(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
        )
    }

def apply_torsion_mmff(nodes, suffix=""):
    """Torsion energy in nodes."""
    return {
        "u%s"
        % suffix: esp.mm.torsion.periodic_torsion_mmff(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
        )
    }

def apply_oop_mmff(nodes, suffix=""):
    """Torsion energy in nodes."""
    return {
        "u%s"
        % suffix: esp.mm.angle.oop_bend_mmff(
            x=nodes.data["x"],
            k=nodes.data["k%s" % suffix],
        )
    }




def apply_improper_torsion(nodes, suffix=""):
    """Improper torsion energy in nodes."""
    if (
        "phases%s" % suffix in nodes.data
        and "periodicity%s" % suffix in nodes.data
    ):
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
        n_multi = nodes.data["k%s" % suffix].shape[-1]
        periodicity=list(range(1, n_multi+1))
        phases=[0.0 for _ in range(n_multi)]
        
        return {
            "u%s"
            % suffix: esp.mm.torsion.periodic_torsion(
                x=nodes.data["x"],
                k=nodes.data["k%s" % suffix],
                phases=phases,
                periodicity=periodicity,
            )
        }


def apply_improper_torsion_mmff(nodes, suffix=""):
    """Improper torsion energy in nodes."""
    if (
        "phases%s" % suffix in nodes.data
        and "periodicity%s" % suffix in nodes.data
    ):
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
        n_multi = nodes.data["k%s" % suffix].shape[-1]
        periodicity=list(range(1, n_multi+1))
        phases=[0.0 for _ in range(n_multi)]
        return {
            "u%s"
            % suffix: esp.mm.torsion.periodic_torsion(
                x=nodes.data["x"],
                k=nodes.data["k%s" % suffix],
                phases=phases,
                periodicity=periodicity,
            )
        }


def apply_bond_gaussian(nodes, suffix=""):
    """Bond energy in nodes."""
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.gaussian_bond(
            x=nodes.data["x"],
            coefficients=nodes.data["coefficients%s" % suffix],
        )
    }


def apply_bond_linear_mixture(nodes, suffix="", phases=[0.0, 1.0]):
    """Bond energy in nodes."""
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.bond.linear_mixture_bond(
            x=nodes.data["x"],
            coefficients=nodes.data["coefficients%s" % suffix],
            phases=phases,
        )
    }


def apply_angle_linear_mixture(nodes, suffix="", phases=[0.0, 1.0]):
    """Bond energy in nodes."""
    # if suffix == '_ref':
    return {
        "u%s"
        % suffix: esp.mm.angle.linear_mixture_angle(
            x=nodes.data["x"],
            coefficients=nodes.data["coefficients%s" % suffix],
            phases=phases,
        )
    }


# =============================================================================
# ENERGY IN HYPERNODES---NONBONDED
# =============================================================================
def apply_nonbonded(nodes, scaling=1.0, suffix=""):
    """Nonbonded in nodes."""
    # TODO: should this be 9-6 or 12-6?
    return {
        "u%s"
        % suffix: scaling
        * esp.mm.nonbonded.lj_12_6(
            x=nodes.data["x"],
            sigma=nodes.data["sigma%s" % suffix],
            epsilon=nodes.data["epsilon%s" % suffix],
        )
    }


def apply_coulomb(nodes, scaling=1.0, suffix=""):
    return {
        "u%s"
        % suffix: scaling
        * esp.mm.nonbonded.coulomb(
            x=nodes.data["x"],
            q=nodes.data["q"],
        )
    }


# =============================================================================
# ENERGY IN GRAPH
# =============================================================================
def energy_in_graph(
    g,
    suffix="",
    terms=["n2", "n3", "n4"],
):  # "onefour", "nonbonded"]):
    """Calculate the energy of a small molecule given parameters and geometry.

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
    import dgl

    if "n2" in terms:
        # apply energy function

        if "coefficients%s" % suffix in g.nodes["n2"].data:
            g.apply_nodes(
                lambda node: apply_bond_linear_mixture(
                    node, suffix=suffix, phases=[1.5, 6.0]
                ),
                ntype="n2",
            )
        else:
            g.apply_nodes(
                lambda node: apply_bond(node, suffix=suffix),
                ntype="n2",
            )

    if "n3" in terms:
        if "coefficients%s" % suffix in g.nodes["n3"].data:
            import math

            g.apply_nodes(
                lambda node: apply_angle_linear_mixture(
                    node, suffix=suffix, phases=[0.0, math.pi]
                ),
                ntype="n3",
            )
        else:
            g.apply_nodes(
                lambda node: apply_angle(node, suffix=suffix),
                ntype="n3",
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

    # if g.number_of_nodes("nonbonded") > 0 and "nonbonded" in terms:
    #     g.apply_nodes(
    #         lambda node: apply_nonbonded(node, suffix=suffix),
    #         ntype="nonbonded",
    #     )

    # if g.number_of_nodes("onefour") > 0 and "onefour" in terms:
    #     g.apply_nodes(
    #         lambda node: apply_nonbonded(
    #             node,
    #             suffix=suffix,
    #             scaling=0.5,
    #         ),
    #         ntype="onefour",
    #     )

    if "nonbonded" in terms or "onefour" in terms:
        esp.mm.nonbonded.multiply_charges(g)

    if "nonbonded" in terms and g.number_of_nodes("nonbonded") > 0:
        g.apply_nodes(
            lambda node: apply_coulomb(
                node,
                suffix=suffix,
                scaling=1.0,
            ),
            ntype="nonbonded",
        )

    if "onefour" in terms and g.number_of_nodes("onefour") > 0:
        g.apply_nodes(
            lambda node: apply_coulomb(
                node,
                suffix=suffix,
                # scaling=0.5,
                scaling=0.8333333333333334,
            ),
            ntype="onefour",
        )

    # sum up energy
    # bonded
    g.multi_update_all(
        {
            "%s_in_g"
            % term: (
                dgl.function.copy_u(u="u%s" % suffix, out="m_%s" % term),
                dgl.function.sum(
                    msg="m_%s" % term, out="u_%s%s" % (term, suffix)
                ),
            )
            for term in terms
            if "u%s" % suffix in g.nodes[term].data
        },
        cross_reducer="sum",
    )

    g.apply_nodes(
        lambda node: {
            "u%s"
            % suffix: sum(
                node.data["u_%s%s" % (term, suffix)]
                for term in terms
                if "u_%s%s" % (term, suffix) in node.data
            )
        },
        ntype="g",
    )

    if "u0" in g.nodes["g"].data:
        g.apply_nodes(
            lambda node: {"u": node.data["u"] + node.data["u0"]},
            ntype="g",
        )

    return g



# =============================================================================
# ENERGY IN GRAPH
# =============================================================================
def energy_in_graph_mmff(
    g,
    suffix="",
    terms=["n2", "n3", "n4"],
):  # "onefour", "nonbonded"]):
    """Calculate the energy of a small molecule given parameters and geometry.

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
    import dgl

    if "n2" in terms:
        # apply energy function

        # if "coefficients%s" % suffix in g.nodes["n2"].data:
        #     g.apply_nodes(
        #         lambda node: apply_bond_linear_mixture(
        #             node, suffix=suffix, phases=[1.5, 6.0]
        #         ),
        #         ntype="n2",
        #     )
        # else:
        g.apply_nodes(
            lambda node: apply_bond_mmff(node, suffix=suffix),
            ntype="n2",
        )

    if "n3" in terms:
        # if "coefficients%s" % suffix in g.nodes["n3"].data:
        #     import math

        #     g.apply_nodes(
        #         lambda node: apply_angle_linear_mixture(
        #             node, suffix=suffix, phases=[0.0, math.pi]
        #         ),
        #         ntype="n3",
        #     )
        # else:
        g.apply_nodes(
            lambda node: apply_angle_mmff(node, suffix=suffix),
            ntype="n3",
        )

        # copy n2 into n3
        
        ijk = g.nodes['n3'].data['idxs']
        ij = ijk[:, :2]
        


        # Extract x and eq by indexing `g.nodes['n2']`
        mask = torch.all(torch.eq(g.nodes['n2'].data['idxs'][:, None, :], ij), dim=-1)
        ij_idx = torch.argmax(mask.int(), dim=0)
        eq2_ij = g.nodes['n2'].data['eq'][ij_idx]
        x_ij = g.nodes['n2'].data['x'][ij_idx]


        kj = torch.roll(ijk, 1, 1)[:, :2]
        mask = torch.all(torch.eq(g.nodes['n2'].data['idxs'][:, None, :], kj), dim=-1)      
        kj_idx = torch.argmax(mask.int(), dim=0)
        eq2_kj = g.nodes['n2'].data['eq'][kj_idx]
        x_kj = g.nodes['n2'].data['x'][kj_idx]

        g.nodes['n3'].data['eq2_ij'] = eq2_ij
        g.nodes['n3'].data['eq2_kj'] = eq2_kj

        g.nodes['n3'].data['x2_ij'] = x_ij
        g.nodes['n3'].data['x2_kj'] = x_kj

        g.apply_nodes(
            lambda node: apply_stretch_bend(node, suffix=suffix),
            ntype="n3",
        )
        
    if g.number_of_nodes("n4") > 0 and "n4" in terms:
        g.apply_nodes(
            lambda node: apply_torsion_mmff(node, suffix=suffix),
            ntype="n4",
        )

    
    if g.number_of_nodes("n4_improper") > 0 and "n4_improper" in terms:
        g.apply_nodes(
            lambda node: apply_improper_torsion(node, suffix=suffix),
            ntype="n4_improper",
        )
    
    if g.number_of_nodes('n4_oop') > 0 and "n4_oop" in terms:
        g.apply_nodes(
            lambda node: apply_oop_mmff(node, suffix=suffix),
            ntype="n4_oop",
        )

    if "nonbonded" in terms or "onefour" in terms:
        esp.mm.nonbonded.multiply_charges(g)

    if "nonbonded" in terms and g.number_of_nodes("nonbonded") > 0:
        g.apply_nodes(
            lambda node: apply_coulomb(
                node,
                suffix=suffix,
                scaling=1.0,
            ),
            ntype="nonbonded",
        )

    if "onefour" in terms and g.number_of_nodes("onefour") > 0:
        g.apply_nodes(
            lambda node: apply_coulomb(
                node,
                suffix=suffix,
                # scaling=0.5,
                scaling=0.8333333333333334,
            ),
            ntype="onefour",
        )

    # sum up energy
    # bonded
    g.multi_update_all(
        {
            "%s_in_g"
            % term: (
                dgl.function.copy_u(u="u%s" % suffix, out="m_%s" % term),
                dgl.function.sum(
                    msg="m_%s" % term, out="u_%s%s" % (term, suffix)
                ),
            )
            for term in terms
            if "u%s" % suffix in g.nodes[term].data
        },
        cross_reducer="sum",
    )

    
    g.apply_nodes(
        lambda node: {
            "u%s"
            % suffix: sum(
                node.data["u_%s%s" % (term, suffix)]
                for term in terms
                if "u_%s%s" % (term, suffix) in node.data
            )
        },
        ntype="g",
    )

    
    if "u0" in g.nodes["g"].data:
        g.apply_nodes(
            lambda node: {"u": node.data["u"] + node.data["u0"]},
            ntype="g",
        )
    
    return g


def energy_in_graph_ii(
    g,
    suffix="",
):
    if g.number_of_nodes("n3") > 0:

        g.apply_nodes(
            lambda node: apply_angle_ii(node, suffix=suffix),
            ntype="n3",
        )

        g.apply_nodes(
            lambda node: {
                "u%s" % suffix: node.data["u%s" % suffix]
                + node.data["u_urey_bradley%s" % suffix]
                + node.data["u_bond_bond%s" % suffix]
                + node.data["u_bond_angle%s" % suffix]
            },
            ntype="n3",
        )

    if g.number_of_nodes("n4") > 0:
        g.apply_nodes(
            lambda node: apply_torsion_ii(node, suffix=suffix),
            ntype="n4",
        )

        g.apply_nodes(
            lambda node: {
                "u%s" % suffix: node.data["u%s" % suffix]
                + node.data["u_angle_angle%s" % suffix]
                + node.data["u_angle_torsion%s" % suffix]
                + node.data["u_angle_angle_torsion%s" % suffix]
                + node.data["u_bond_torsion%s" % suffix]
            },
            ntype="n4",
        )

    return g


class EnergyInGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(EnergyInGraph, self).__init__()
        self.args = args
        self.kwargs = kwargs
        

    def forward(self, g):
        return energy_in_graph(g, *self.args, **self.kwargs)


class EnergyInGraphII(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(EnergyInGraphII, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return energy_in_graph_ii(g, *self.args, **self.kwargs)


class EnergyInGraphMMFF(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(EnergyInGraphMMFF, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return energy_in_graph_mmff(g, *self.args, **self.kwargs)


class CarryII(torch.nn.Module):
    def forward(self, g):
        import math

        import dgl

        g.multi_update_all(
            {
                "n2_as_0_in_n3": (
                    dgl.function.copy_u("u", "m_u_0"),
                    dgl.function.sum("m_u_0", "u_left"),
                ),
                "n2_as_1_in_n3": (
                    dgl.function.copy_u("u", "m_u_1"),
                    dgl.function.sum("m_u_1", "u_right"),
                ),
                "n2_as_0_in_n4": (
                    dgl.function.copy_u("u", "m_u_0"),
                    dgl.function.sum("m_u_0", "u_bond_left"),
                ),
                "n2_as_1_in_n4": (
                    dgl.function.copy_u("u", "m_u_1"),
                    dgl.function.sum("m_u_1", "u_bond_center"),
                ),
                "n2_as_2_in_n4": (
                    dgl.function.copy_u("u", "m_u_2"),
                    dgl.function.sum("m_u_2", "u_bond_right"),
                ),
                "n3_as_0_in_n4": (
                    dgl.function.copy_u("u", "m3_u_0"),
                    dgl.function.sum("m3_u_0", "u_angle_left"),
                ),
                "n3_as_1_in_n4": (
                    dgl.function.copy_u("u", "m3_u_1"),
                    dgl.function.sum("m3_u_1", "u_angle_right"),
                ),
            },
            cross_reducer="sum",
        )

        return g
