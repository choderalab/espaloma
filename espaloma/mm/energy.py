# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import espaloma as esp

# =============================================================================
# ENERGY IN HYPERNODES---BONDED
# =============================================================================
def apply_bond(nodes, suffix=''):
    """ Bond energy in nodes. """
    return {
        'u%s' % suffix: esp.mm.bond.harmonic_bond(
            x=nodes.data['x'],
            k=nodes.data['k%s' % suffix],
            eq=nodes.data['eq%s' % suffix],
        )
    }

def apply_angle(nodes, suffix=''):
    """ Angle energy in nodes. """
    return {
        'u%s' % suffix: esp.mm.angle.harmonic_angle(
            x=nodes.data['x'],
            k=nodes.data['k%s' % suffix],
            eq=nodes.data['eq%s' % suffix],
        )
    }

def apply_torsion(nodes, suffix=''):
    """ Torsion energy in nodes. """
    return {
        'u%s' % suffix: esp.mm.torsion.periodic_torsion(
            x=nodes.data['x'],
            k=nodes.data['k%s' % suffix],
            eq=nodes.data['eq%s' % suffix],
        )
    }

# =============================================================================
# ENERGY IN HYPERNODES---NONBONDED
# =============================================================================
def apply_nonbonded(nodes, suffix=''):
    """ Nonbonded in nodes. """
    return {
        'u%s' % suffix: esp.mm.nonbonded.lj_12_6(
            x=nodes.data['x'],
            sigma=nodes.data['sigma%s' % suffix],
            epsilon=nodes.data['epsilon%s' % suffix],
        )
    }


# =============================================================================
# ENERGY IN GRAPH
# =============================================================================
def energy_in_graph(g, suffix=''):
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
    esp.mm.nonbonded.lorentz_berthelot(g, suffix=suffix)

    # apply energy function
    g.apply_nodes(
            lambda node: apply_bond(node, suffix=suffix), 
            ntype='n2'
    )


    g.apply_nodes(
            lambda node: apply_angle(node, suffix=suffix),
            ntype='n3',
    )

    if g.number_of_nodes('nonbonded') > 0:
        g.apply_nodes(
                lambda node: apply_nonbonded(node, suffix=suffix), 
                ntype='nonbonded'
        )

    if g.number_of_nodes('onefour') > 0:
        g.apply_nodes(
                lambda node: apply_nonbonded(node, suffix=suffix), 
                ntype='onefour'
        )

    # sum up energy
    # bonded
    g.multi_update_all(
        {
            **{
                'n%s_in_g' % idx: (
                    dgl.function.copy_src(src='u%s' % suffix, out='m_%s' % idx),
                    dgl.function.sum(msg='m_%s' % idx, out='u%s%s' % (idx, suffix))
                ) for idx in [2, 3]
            },
            **{
                '%s_in_g' % term: (
                    dgl.function.copy_src(src='u%s' % suffix, out='m_%s' % term),
                    dgl.function.sum(msg='m_%s' % term, out='u_%s%s' % (term, suffix))
                ) for term in ['onefour', 'nonbonded']
            },
        },
        'sum'
    )

    g.apply_nodes(
        lambda node: {
            'u%s' % suffix: node.data['u2%s' % suffix] + node.data['u3%s' % suffix] 
             # + node.data['onefour'] + node.data['nonbonded']
        },
        ntype='g'
    )

    return g
