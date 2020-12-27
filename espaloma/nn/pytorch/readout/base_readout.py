# =============================================================================
# IMPORTS
# =============================================================================
import abc

import torch


# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseReadout(abc.ABC, torch.nn.Module):
    """ Base class for readout function.

    """

    def __init__(self):
        super(BaseReadout, self).__init__()

    @abc.abstractmethod
    def forward(self, g, x=None, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, g, x, *args, **kwargs):
        raise NotImplementedError
