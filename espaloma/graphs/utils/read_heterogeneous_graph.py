""" Build heterogeneous graph from homogeneous ones.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def heterogeneous_graph_from_homogeneous(g):
    r""" Build heterogeneous graph from homogeneous ones.

    
    Note
    ----
    For now we name single node, two-, three, and four-,
    hypernodes as `n1`, `n2`, `n3`, and `n4`. These correspond
    to atom, bond, angle, and torsion nodes in chemical graphs.


    Parameters
    ----------
    g : `espaloma.HomogeneousGraph` object
        the homogeneous graph to be translated.

    Returns
    -------
    hg : `espaloma.HeterogeneousGraph` object
        the resulting heterogeneous graph.

    """

    # initialize empty dictionary
    hg = {}

    # NOTE:
    # here we only define the neighboring relationship
    # on atom level
    for idx in range(1, 5):
        hg[(
            'n%s' % idx,
            'n%s_neighbors_n%s' % (idx, idx)
            'n%s' % idx
            )] = []

    
    # NOTE:
    # here we define all the possible
    # 'has' and 'in' relationships.
    # TODO:
    # we'll test later to see if this adds too much overhead
    for small_idx in range(1, 5):
        for big_idx in range(small_idx+1, 5):
            for pos_idx in range(big_idx - small_idx):
                hg[(
                    'n%s' % small_idx,
                    'n%s_as_%s_in_n%s' % (small_idx, pos_idx, big_idx),
                    'n%s' % big_idx
                    )] = []

                hg[(
                    'n%s' % big_idx,
                    'n%s_has_%s_n%s' % (big_idx, pos_idx, small_idx),
                    'n%s' % small_idx
                   )] = []





