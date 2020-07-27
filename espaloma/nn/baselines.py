# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class FreeParameterBaseline(torch.nn.Module):
    """ Parametrize a graph by populating the parameters with free
    `torch.nn.Parameter`.


    """
    def __init__(self, g_ref):
        super(FreeParameterBaseline, self).__init__()
        self.g_ref = g_ref

        # whenever there is a reference parameter,
        # assign a `torch.nn.Parameter`
        for term in self.g_ref.ntypes:
            for param, param_value in self.g_ref.nodes[term].data.items():
                if param.endswith("_ref"):
                    setattr(
                        self,
                        "%s_%s" % (term, param),
                        torch.nn.Parameter(
                            torch.zeros_like(
                                param_value
                            )
                        )
                    )

    def forward(self, g):
        for term in self.g_ref.ntypes:
            for param, param_value in self.g_ref.nodes[term].data.items():
                if param.endswith("_ref"):
                    g.nodes[term].data[param.replace("_ref", "")] = getattr(
                        self,
                        "%s_%s" % (term, param),
                    )

        
        return g




