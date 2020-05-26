""" Legacy models from DGL.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import math
import dgl
from dgl.nn import pytorch as dgl_pytorch

# =============================================================================
# MODULE CLASSES
# =============================================================================


class GN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_name="GraphConv",
        kwargs={},
    ):
        super(GN, self).__init__()

        self.gn = getattr(dgl_pytorch.conv, model_name)(
                in_features, 
                out_features, 
                **kwargs)

        # register these properties here for downstream handling
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, g):
        x = g.ndata["h"]
        x = self.gn(g, x)
        g.ndata["h"] = x
        return g

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def gn(model_name="GraphConv", kwargs={}):

    return lambda in_features, out_features: GN(
            in_features=in_features, 
            out_features=out_features, 
            model_name=model_name, 
            kwargs=kwargs)
