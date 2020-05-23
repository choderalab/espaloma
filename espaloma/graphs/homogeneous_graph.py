# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
import abc
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HomogeneousGraph(espaloma.graph, dgl.Graph):
    """ Homogeneous graph that contains no more than connectivity and
    atom attributes.


    """
    def __init__(self):
        super(HomogeneousGraph, self).__init__()


