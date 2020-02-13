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
        g_param,
        g_geo,
        one_four_scaling=0.5,
        switch=0.1,
        damping=0.0):
    """ Calculate energy based on two graphs, one with geometry, one with
    parameters.

    """
    scaling = {'one_four': one_four_scaling, 'nonbonded': 1.0}

    for term in ['bond', 'angle', 'torsion']:
        x = g_geo.nodes[term].data['x']
        k = g_param.nodes[term].data['k']
        eq = g_param.nodes[term].data['eq']
        u = getattr(
            hgfp.mm.energy,
            term)(x, k, eq)
        g_param.nodes[term].data['energy'] = u

    for term in ['one_four', 'nonbonded']:

        x = g_geo.nodes[term].data['x']
        sigma_pair = g_param.nodes[term].data['sigma_pair']
        epsilon_pair = g_param.nodes[term].data['epsilon_pair']
        u = scaling[term] * hgfp.mm.energy.lj(x, sigma_pair, epsilon_pair)
        g_param.nodes[term].data['energy'] = u

    return g_param
