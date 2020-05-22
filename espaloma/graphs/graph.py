# =============================================================================
# IMPORTS
# =============================================================================

import espaloma
import abc

class Graph(abc.ABC):
    """ Base class of various graph objects that we host data in.
    
    """

    @property
    @abstractmethod
    def _stage(self):
        pass

