# =============================================================================
# IMPORTS
# =============================================================================
import torch
from .base_readout import BaseReadout


# =============================================================================
# MODULE CLASSES
# =============================================================================
class NodeTyping(BaseReadout):
    """ Simple typing on homograph.

    """

    def __init__(self, in_features, n_classes=100):
        super(NodeTyping, self).__init__()
        self.c = torch.nn.Linear(in_features, n_classes)

    def forward(self, g):
        g.apply_nodes(
            ntype="n1",
            func=lambda node: {"nn_typing": self.c(node.data["h"])},
        )
        return g
