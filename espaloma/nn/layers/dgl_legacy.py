""" Legacy models from DGL.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import math
import dgl
from dgl.nn import pytorch as dgl_pytorch
from copy import deepcopy

# =============================================================================
# CONSTANT
# =============================================================================
DEFAULT_MODEL_KWARGS = {
    "SAGEConv": {"aggregator_type": "mean"},
    "GATConv": {"num_heads": 4},
    "TAGConv": {"k": 2},
}


# =============================================================================
# MODULE CLASSES
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
        self, in_features, out_features, model_name="GraphConv", kwargs={},
    ):
        super(GN, self).__init__()

        if kwargs == {}:
            if model_name in DEFAULT_MODEL_KWARGS:
                kwargs = DEFAULT_MODEL_KWARGS[model_name]

        self.gn = getattr(dgl_pytorch.conv, model_name)(
            in_features, out_features, **kwargs
        )

        # register these properties here for downstream handling
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, g, x):
        return self.gn(g, x)


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def gn(model_name="GraphConv", kwargs={}):
    if model_name == "GINConv":
        return lambda in_features, out_features: dgl_pytorch.conv.GINConv(
            apply_func=torch.nn.Linear(in_features, out_features), aggregator_type="sum"
        )

    else:
        return lambda in_features, out_features: GN(
            in_features=in_features,
            out_features=out_features,
            model_name=model_name,
            kwargs=kwargs,
        )
