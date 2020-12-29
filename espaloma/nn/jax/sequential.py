""" Chain mutiple layers of GN together.
"""
import dgl
import jax
from flax import linen as nn

class _Sequential(nn.Module):
    """ Sequentially staggered neural networks.

    """

    layer: nn.Module
    config: list
    in_features: int
    from typing import Union
    model_kwargs: Union[dict, None] = None

    def setup(self):
        if self.model_kwargs == None:
            self.model_kwargs = {}

        self.exes = []

        dim = self.in_features

        for idx, exe in enumerate(self.config):
            try:
                exe = float(exe)
                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), self.layer(dim, exe, **self.model_kwargs))

                dim = exe
                self.exes.append("d" + str(idx))

            # str -> activation
            elif isinstance(exe, str):
                activation = getattr(nn, exe)

                setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))


    def __call__(self, g, x):
        for exe in self.exes:
            if exe.startswith("d"):
                if g is not None:
                    x = getattr(self, exe)(g, x)
                else:
                    x = getattr(self, exe)(x)
            else:
                x = getattr(self, exe)(x)

        return x


class Sequential(nn.Module):
    """ Sequential neural network with input layers.

    """

    layer: nn.Module
    config: list
    feature_units: int = 117
    input_units: int = 128
    from typing import Union
    model_kwargs: Union[None, dict] = None

    def setup(self):
        if self.model_kwargs == None:
            self.model_kwargs = {}

        self._sequential = _Sequential(
            self.layer, self.config, in_features=self.input_units, model_kwargs=self.model_kwargs
        )

    def _forward(self, g, x):
        """ Forward pass with graph and features.

        """
        for exe in self.exes:
            if exe.startswith("d"):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        return x

    @nn.compact
    def __call__(self, g, x=None):
        """ Forward pass.

        Parameters
        ----------
        g : `dgl.DGLHeteroGraph`,
            input graph

        Returns
        -------
        g : `dgl.DGLHeteroGraph`
            output graph
        """

        # get homogeneous subgraph
        g_ = dgl.to_homo(g.edge_type_subgraph(["n1_neighbors_n1"]))

        if x is None:
            # get node attributes
            x = g.nodes["n1"].data["h0"]
            x = nn.tanh(nn.Dense(self.input_units)(x))

        # message passing on homo graph
        x = self._sequential(g_, x)

        # put attribute back in the graph
        g.nodes["n1"].data["h"] = x

        return g
