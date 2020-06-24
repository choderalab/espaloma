""" Chain mutiple layers of GN together.
"""
import torch
import dgl


class Sequential(torch.nn.Module):
    def __init__(
        self,
        layer,
        config,
        feature_units=117,
        input_units=128,
        model_kwargs={},
    ):
        super(Sequential, self).__init__()

        # the initial dimensionality
        dim = input_units

        # record the name of the layers in a list
        self.exes = []

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))

                dim = exe
                self.exes.append("d" + str(idx))

            # str -> activation
            elif isinstance(exe, str):
                activation = getattr(torch.nn.functional, exe)

                setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))

            # float -> dropout
            elif isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def _forward(self, g, x):
        """ Forward pass with graph and features.

        """
        for exe in self.exes:
            if exe.startswith('d'):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        return x

    def forward(self, g, x=None):
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
        g_ = dgl.to_homo(
            g.edge_type_subgraph(['n1_neighbors_n1']))

        if x is None:
            # get node attributes
            x = g.nodes['n1'].data['h0']
            x = self.f_in(x)

        # message passing on homo graph
        x = self._forward(g_, x)

        # put attribute back in the graph
        g.nodes['n1'].data['h'] = x

        return g
