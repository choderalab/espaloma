# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def batch(graphs):
    r""" Batch multiple heterogenous graph.

    Parameters
    ----------
    graphs : list of `esp.HeterogeneousGraph`

    """

    # extract dgl graphs 
    _graphs = [graph._g for graph in graphs]

    # batch dgl graph
    _graphs = dgl.batch_hetero(_graphs)

    # put batched graph back into HeterogeneousGraph object
    
    # TODO: allow more stuff to be done here
    return HeterogeneousGraph(dgl_hetero_graph = _graphs)


def unbatch(graph):
    r""" Unbatch multiple heterogeneous graph.

    """

    # extract dgl graph
    _graphs = graph._g

    # unbatch dgl graph
    _graphs = dgl.unbatch_hetero(_graphs)

    return [HeterogeneousGraph(dgl_hetero_graph=g) for g in _graphs]


# =============================================================================
# MODULE CLASSES
# =============================================================================
class HeterogeneousGraph(esp.Graph):
    r""" Homogeneous graph that contains no more than connectivity and
    atom attributes.

    Note
    ----    
    This module is not currently in use as subclassing
    `dgl.DGLHeteroGraph` is not encouraged.

    Parameters
    ----------
    homogeneous_graph: `espaloma.HomogeneousGraph` object


    """

    def __init__(
            self, 
            homogeneous_graph=None,
            dgl_hetero_graph=None,
            mol=None,
        ):

        if homogeneous_graph is not None:
            self._g = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous(
                homogeneous_graph
            )

        elif dgl_hetero_graph is not None:
            self._g = dgl_hetero_graph

        elif mol is not None:
            homogeneous_graph = esp.HomogeneousGraph(mol)
            self._g = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous(
                homogeneous_graph
            )
        
    @property
    def _stage(self):
        return "heterogeneous"
