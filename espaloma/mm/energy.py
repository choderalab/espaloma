# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import espaloma as esp

# =============================================================================
# ENERGY IN HYPERNODES---BONDED
# =============================================================================
def apply_bond(nodes):
    """ Bond energy in nodes. """
    return {
        'u': esp.mm.bond.harmonic_bond(
            x=nodes.data['x'],
            k=nodes.data['k'],
            eq=nodes.data['eq'],
        )
    }

def apply_angle(nodes):
    """ Angle energy in nodes. """
    return {
        'u': esp.mm.angle.harmonic_angle(
            x=nodes.data['x'],
            k=nodes.data['k'],
            eq=nodes.data['eq'],
        )
    }

def apply_torsion(nodes):
    """ Torsion energy in nodes. """
    return {
        'u': esp.mm.torsion.periodic_torsion(
            x=nodes.data['x'],
            k=nodes.data['k'],
            eq=nodes.data['eq'],
        )
    }

# =============================================================================
# ENERGY IN HYPERNODES---NONBONDED
# =============================================================================
def apply_nonbonded(nodes):
    """ Nonbonded in nodes. """
    return {
        'u': esp.mm.nonbonded.lj_12_6(
            x=nodes.data['x'],
            sigma=nodes.data['sigma'],
            epsilon=nodes.data['epsilon'],
        )
    }


# =============================================================================
# ENERGY IN GRAPH
# =============================================================================
def energy_in_graph(g):
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

    # apply combination rule
    esp.mm.nonbonded.lorentz_berthelot(g)

    # apply energy function
    g.apply_nodes(apply_bond, ntype='n2')
    g.apply_nodes(apply_angle, ntype='n3')
    # g.apply_nodes(apply_torsion, ntype='n4')

    if g.number_of_nodes('nonbonded') > 0:
        g.apply_nodes(apply_nonbonded, ntype='nonbonded')

    if g.number_of_nodes('onefour') > 0:
        g.apply_nodes(apply_nonbonded, ntype='onefour')

    # sum up energy
    # bonded
    g.multi_update_all(
        {
            **{
                'n%s_in_g' % idx: (
                    dgl.function.copy_src(src='u', out='m_%s' % idx),
                    dgl.function.sum(msg='m_%s' % idx, out='u%s' % idx)
                ) for idx in [2, 3]
            },
            **{
                '%s_in_g' % term: (
                    dgl.function.copy_src(src='u', out='m_%s' % term),
                    dgl.function.sum(msg='m_%s' % term, out='u_%s' % term)
                ) for term in ['onefour', 'nonbonded']
            },
        },
        'sum'
    )

    g.apply_nodes(
        lambda node: {
            'u': node.data['u2'] + node.data['u3'] # + node.data['onefour'] + node.data['nonbonded']
        },
        ntype='g'
    )

    return g
