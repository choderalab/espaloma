# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import espaloma as esp

# =============================================================================
# MODULE CLASSES
# =============================================================================
class GraphLevelReadout(torch.nn.Module):
    """ Readout from graph level. """

    def __init__(
        self,
        in_features,
        config_local,
        config_global,
        out_name,
        pool=dgl.function.sum,
    ):

        super(GraphLevelReadout, self).__init__()
        self.in_features = in_features
        self.config_local = config_local
        self.config_global = config_global
        self.d_local = esp.nn.sequential._Sequential(
            in_features=in_features,
            config=config_local,
            layer=torch.nn.Linear,
        )

        mid_features = [x for x in config_local if isinstance(x, int)][-1]

        self.d_global = esp.nn.sequential._Sequential(
            in_features=mid_features,
            config=config_global,
            layer=torch.nn.Linear,
        )

        self.pool = pool
        self.out_name = out_name

    def forward(self, g):
        g.apply_nodes(
            lambda node: {"h_global": self.d_local(None, node.data["h"])},
            ntype="n1",
        )

        g.update_all(
            dgl.function.copy_src("h_global", "m"),
            self.pool("m", "h_global"),
            etype="n1_in_g",
        )

        g.apply_nodes(
            lambda node: {
                self.out_name: self.d_global(None, node.data["h_global"])
            },
            ntype="g",
        )

        return g
