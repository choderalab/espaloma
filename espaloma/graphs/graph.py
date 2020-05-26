# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Graph(abc.ABC):
    """ Base class of various graph objects that we host data in.
    
    """

    @property
    @abc.abstractmethod
    def _stage(self):
        pass

