# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import espaloma as esp

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
class NodeTyping(torch.nn.Module):
    """ Simple typing on homograph.

    """
    def __init__(self, in_features, n_classes):
        super(NodeTyping, self).__init__()
        self.c = torch.nn.Linear(in_features, n_classes)

    def forward(self, g):
        g.apply_nodes(
                lambda node: {'nn_typing': self.c(node.data['h'])})
        return g
