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

    _stage = {
            'type': 'base',
            'batched': False,
            'neuralized': False,
            'legacy_typed': False,
            'has_coordinate': False
        }

    @property
    def stage(self):
        return _stage

    def set_stage(self, **kwargs):
        for key, value in kwargs.items():
            self._stage[key] = value


