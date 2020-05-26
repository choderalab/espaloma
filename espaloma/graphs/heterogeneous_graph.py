# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HeterogeneousGraph(esp.Graph, dgl.DGLHeteroGraph):
    r""" Homogeneous graph that contains no more than connectivity and
    atom attributes.

    Parameters
    ----------
    mol : a `rdkit.Chem.Molecule` or `openeye.GraphMol` object

    """
    def __init__(self, homogeneous_graph=None):
        # super(HeterogeneousGraph, self).__init__()
        
        if homogeneous_graph is not None:
            self = esp.graphs.utils.read_heterogeneous_graph\
                    .heterogeneous_graph_from_homogeneous(
                            homogeneous_graph)


        
    @property
    def _stage(self):
        return 'heterogeneous'



