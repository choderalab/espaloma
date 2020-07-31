# =============================================================================
# IMPORTS
# =============================================================================
import torch


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
                if param.endswith("_ref") and "u" not in param:
                    setattr(
                        self,
                        "%s_%s" % (term, param.replace("_ref", "")),
                        torch.nn.Parameter(
                            torch.zeros_like(param_value.clone().detach(),)
                        ),
                    )

    def forward(self, g):
        update_dicts = {node: {} for node in self.g_ref.ntypes}

        for term in self.g_ref.ntypes:
            for param, param_value in self.g_ref.nodes[term].data.items():
                if param.endswith("_ref"):
                    if hasattr(
                        self, "%s_%s" % (term, param.replace("_ref", ""))
                    ):

                        update_dicts[term][
                            param.replace("_ref", "")
                        ] = getattr(
                            self, "%s_%s" % (term, param.replace("_ref", "")),
                        )

        for node, update_dict in update_dicts.items():
            for param, param_value in update_dict.items():
                g.nodes[node].data[param] = param_value

        return g
