""" Legacy models from DGL.

"""

import math
from copy import deepcopy

import dgl

# =============================================================================
# IMPORTS
# =============================================================================
import torch
from dgl.nn import jax as dgl_jax

import jax
from flax import linen as nn

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
class GN(nn.Module):
    in_features: int
    out_features: int
    model_name: str = "GraphConv"
    from typing import Union
    kwargs: Union[dict, None] = None

    def setup(self):
        if self.kwargs == None:
            self.kwargs = {}

        if self.kwargs == {}:
            if self.model_name in DEFAULT_MODEL_KWARGS:
                self.kwargs = DEFAULT_MODEL_KWARGS[self.model_name]

        self.gn = getattr(dgl_jax.conv, self.model_name)(
            self.in_features, self.out_features, **self.kwargs
        )

    def __call__(self, g, x):
        return self.gn(g, x)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def gn(model_name="GraphConv", kwargs={}):
        return lambda in_features, out_features: GN(
            in_features=in_features,
            out_features=out_features,
            model_name=model_name,
            kwargs=kwargs,
        )
