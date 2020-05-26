# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HeterogeneousGraph(esp.Graph):
    r""" Homogeneous graph that contains no more than connectivity and
    atom attributes.

    Parameters
    ----------
    homogeneous_graph: `espaloma.HomogeneousGraph` object


    """
    def __init__(self, homogeneous_graph):
    
        self._graph = esp.graphs.utils.read_heterogeneous_graph\
                .heterogeneous_graph_from_homogeneous(
                        homogeneous_graph)


        
    @property
    def _stage(self):
        return 'heterogeneous'



