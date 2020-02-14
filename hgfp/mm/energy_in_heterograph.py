""" Calculate energy within a graph.

"""
# =============================================================================
# imports
# =============================================================================
import torch
import dgl
import hgfp

# =============================================================================
# module functions
# =============================================================================
def u(
        g,
        one_four_scaling=0.5,
        switch=1.0,
        damping=1e-3):
    """ Calculate energy based on two graphs, one with geometry, one with
    parameters.

    """
    scaling = {'one_four': one_four_scaling, 'nonbonded': 1.0}

    for term in ['bond', 'angle', 'torsion']:
        x = g.nodes[term].data['x']
        k = g.nodes[term].data['k']
        eq = g.nodes[term].data['eq']
        u = getattr(
            hgfp.mm.energy,
            term)(x, k, eq)
        g.nodes[term].data['energy'] = u

    for term in ['one_four', 'nonbonded']:

        x = g.nodes[term].data['x']
        sigma_pair = g.nodes[term].data['sigma_pair']
        epsilon_pair = g.nodes[term].data['epsilon_pair']
        u = scaling[term] * hgfp.mm.energy.lj(x, sigma_pair, epsilon_pair, switch=switch, damping=damping)
        g.nodes[term].data['energy'] = u


    g.multi_update_all(
        {

            '%s_in_mol' % term: (
                dgl.function.copy_src(src='energy', out='m'),
                dgl.function.sum(msg='m', out='u' + term)) for term in [
                        'bond',
                        'angle',
                        'torsion',
                        'one_four',
                        'nonbonded'
                    ]
        },
        'stack')

    return g
