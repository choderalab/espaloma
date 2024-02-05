# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import torch


# =============================================================================
# MODULE CLASSES
# =============================================================================
class JanossyPooling(torch.nn.Module):
    """Janossy pooling (arXiv:1811.01900) to average node representation
    for higher-order nodes.


    """

    def __init__(
        self,
        config,
        in_features,
        out_features={
            1: ["sigma", "epsilon", "q"],
            2: ["k", "eq"],
            3: ["k", "eq"],
            4: ["k", "eq"],
        },
        out_features_dimensions=-1,
        pool=torch.add,
    ):
        super(JanossyPooling, self).__init__()

        # if users specify out features as lists,
        # assume dimensions to be all zero
        for level in out_features.keys():
            if isinstance(out_features[level], list):
                out_features[level] = dict(
                    zip(out_features[level], [1 for _ in out_features[level]])
                )

        # bookkeeping
        self.out_features = out_features
        self.levels = [key for key in out_features.keys() if key != 1]
        self.pool = pool

        # get output features
        mid_features = [x for x in config if isinstance(x, int)][-1]

        # set up networks
        for level in self.levels:

            # set up individual sequential networks
            setattr(
                self,
                "sequential_%s" % level,
                esp.nn.sequential._Sequential(
                    in_features=in_features * level,
                    config=config,
                    layer=torch.nn.Linear,
                ),
            )

            for feature, dimension in self.out_features[level].items():
                setattr(
                    self,
                    "f_out_%s_to_%s" % (level, feature),
                    torch.nn.Linear(
                        mid_features,
                        dimension,
                    ),
                )

        if 1 not in self.out_features:
            return

        # atom level
        self.sequential_1 = esp.nn.sequential._Sequential(
            in_features=in_features, config=config, layer=torch.nn.Linear
        )

        for feature, dimension in self.out_features[1].items():
            setattr(
                self,
                "f_out_1_to_%s" % feature,
                torch.nn.Linear(
                    mid_features,
                    dimension,
                ),
            )

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph,
            input graph.
        """
        import dgl

        # copy
        g.multi_update_all(
            {
                
                "n1_as_%s_in_n%s"
                % (relationship_idx, big_idx): (
                    dgl.function.copy_u("h", "m%s" % relationship_idx),
                    dgl.function.mean(
                        "m%s" % relationship_idx, "h%s" % relationship_idx
                    ),
                )
                for big_idx in self.levels
                for relationship_idx in range(big_idx)
            },
            cross_reducer="sum",
        )

        # pool
        for big_idx in self.levels:

            if g.number_of_nodes("n%s" % big_idx) == 0:
                continue

            g.apply_nodes(
                func=lambda nodes: {
                    feature: getattr(
                        self, "f_out_%s_to_%s" % (big_idx, feature)
                    )(
                        self.pool(
                            getattr(self, "sequential_%s" % big_idx)(
                                None,
                                torch.cat(
                                    [
                                        nodes.data["h%s" % relationship_idx]
                                        for relationship_idx in range(big_idx)
                                    ],
                                    dim=1,
                                ),
                            ),
                            getattr(self, "sequential_%s" % big_idx)(
                                None,
                                torch.cat(
                                    [
                                        nodes.data["h%s" % relationship_idx]
                                        for relationship_idx in range(
                                            big_idx - 1, -1, -1
                                        )
                                    ],
                                    dim=1,
                                ),
                            ),
                        ),
                    )
                    for feature in self.out_features[big_idx].keys()
                },
                ntype="n%s" % big_idx,
            )

        if 1 not in self.out_features:
            return g

        # atom level
        g.apply_nodes(
            func=lambda nodes: {
                feature: getattr(self, "f_out_1_to_%s" % feature)(
                    self.sequential_1(g=None, x=nodes.data["h"])
                )
                for feature in self.out_features[1].keys()
            },
            ntype="n1",
        )

        return g


class JanossyPoolingImproper(torch.nn.Module):
    """Janossy pooling (arXiv:1811.01900) to average node representation
    for improper torsions.


    """

    def __init__(
        self,
        config,
        in_features,
        out_features={
            "k": 2,
        },
        out_features_dimensions=-1,
    ):
        super(JanossyPoolingImproper, self).__init__()

        # if users specify out features as lists,
        # assume dimensions to be all zero

        # bookkeeping
        self.out_features = out_features
        self.levels = ["n4_improper"]

        # get output features
        mid_features = [x for x in config if isinstance(x, int)][-1]

        # set up networks
        for level in self.levels:

            # set up individual sequential networks
            setattr(
                self,
                "sequential_%s" % level,
                esp.nn.sequential._Sequential(
                    in_features=4 * in_features,
                    config=config,
                    layer=torch.nn.Linear,
                ),
            )

            for feature, dimension in self.out_features.items():
                setattr(
                    self,
                    "f_out_%s_to_%s" % (level, feature),
                    torch.nn.Linear(
                        mid_features,
                        dimension,
                    ),
                )

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph,
            input graph.
        """
        import dgl

        # copy
        g.multi_update_all(
            {
                "n1_as_%s_in_%s"
                % (relationship_idx, big_idx): (
                    dgl.function.copy_u("h", "m%s" % relationship_idx),
                    dgl.function.mean(
                        "m%s" % relationship_idx, "h%s" % relationship_idx
                    ),
                )
                for big_idx in self.levels
                for relationship_idx in range(4)
            },
            cross_reducer="sum",
        )

        if g.number_of_nodes("n4_improper") == 0:
            return g

        # pool
        #   sum over three cyclic permutations of "h0", "h2", "h3", assuming "h1" is the central atom in the improper
        #   following the smirnoff trefoil convention [(0, 1, 2, 3), (2, 1, 3, 0), (3, 1, 0, 2)]
        #   https://github.com/openff.toolkit/openff.toolkit/blob/166c9864de3455244bd80b2c24656bd7dda3ae2d/openff.toolkit/typing/engines/smirnoff/parameters.py#L3326-L3360

        ## Set different permutations based on which definition of impropers
        ##  are being used
        permuts = [(0, 1, 2, 3), (2, 1, 3, 0), (3, 1, 0, 2)]
        stack_permuts = lambda nodes, p: torch.cat(
            [nodes.data[f"h{i}"] for i in p], dim=1
        )

        for big_idx in self.levels:
            inner_net = getattr(self, f"sequential_{big_idx}")
            
            g.apply_nodes(
                func=lambda nodes: {
                    feature: getattr(self, f"f_out_{big_idx}_to_{feature}")(
                        torch.sum(
                            torch.stack(
                                [
                                    inner_net(
                                        g=None, x=stack_permuts(nodes, p)
                                    )
                                    for p in permuts
                                ],
                                dim=0,
                            ),
                            dim=0,
                        )
                    )
                    for feature in self.out_features.keys()
                },
                ntype=big_idx,
            )

        return g


class JanossyPoolingWithSmirnoffImproper(torch.nn.Module):
    """Janossy pooling (arXiv:1811.01900) to average node representation
    for improper torsions.
    """

    def __init__(
        self,
        config,
        in_features,
        out_features={
            "k": 2,
        },
        out_features_dimensions=-1,
    ):
        super(JanossyPoolingWithSmirnoffImproper, self).__init__()

        # if users specify out features as lists,
        # assume dimensions to be all zero

        # bookkeeping
        self.out_features = out_features
        self.levels = ["n4_improper"]

        # get output features
        mid_features = [x for x in config if isinstance(x, int)][-1]

        # set up networks
        for level in self.levels:

            # set up individual sequential networks
            setattr(
                self,
                "sequential_%s" % level,
                esp.nn.sequential._Sequential(
                    in_features=4 * in_features,
                    config=config,
                    layer=torch.nn.Linear,
                ),
            )

            for feature, dimension in self.out_features.items():
                setattr(
                    self,
                    "f_out_%s_to_%s" % (level, feature),
                    torch.nn.Linear(
                        mid_features,
                        dimension,
                    ),
                )

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph,
            input graph.
        """
        import dgl

        # copy
        g.multi_update_all(
            {
                "n1_as_%s_in_%s"
                % (relationship_idx, big_idx): (
                    dgl.function.copy_u("h", "m%s" % relationship_idx),
                    dgl.function.mean(
                        "m%s" % relationship_idx, "h%s" % relationship_idx
                    ),
                )
                for big_idx in self.levels
                for relationship_idx in range(4)
            },
            cross_reducer="sum",
        )

        if g.number_of_nodes("n4_improper") == 0:
            return g

        # pool
        #   sum over three cyclic permutations of "h0", "h2", "h3", assuming "h1" is the central atom in the improper
        #   following the smirnoff trefoil convention [(0, 1, 2, 3), (2, 1, 3, 0), (3, 1, 0, 2)]
        #   https://github.com/openff.toolkit/openff.toolkit/blob/166c9864de3455244bd80b2c24656bd7dda3ae2d/openff.toolkit/typing/engines/smirnoff/parameters.py#L3326-L3360

        # TODO to check
        ## Set different permutations based on which definition of impropers
        ##  are being used
        permuts = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
        stack_permuts = lambda nodes, p: torch.cat(
            [nodes.data[f"h{i}"] for i in p], dim=1
        )

        for big_idx in self.levels:
            inner_net = getattr(self, f"sequential_{big_idx}")

            g.apply_nodes(
                func=lambda nodes: {
                    feature: getattr(self, f"f_out_{big_idx}_to_{feature}")(
                        torch.sum(
                            torch.stack(
                                [
                                    inner_net(
                                        g=None, x=stack_permuts(nodes, p)
                                    )
                                    for p in permuts
                                ],
                                dim=0,
                            ),
                            dim=0,
                        )
                    )
                    for feature in self.out_features.keys()
                },
                ntype=big_idx,
            )

        return g


class JanossyPoolingNonbonded(torch.nn.Module):
    """Janossy pooling (arXiv:1811.01900) to average node representation
    for nonbonded interactions.


    """

    def __init__(
        self,
        config,
        in_features,
        out_features={"sigma": 1, "epsilon": 1},
        out_features_dimensions=-1,
    ):
        super(JanossyPoolingNonbonded, self).__init__()

        # if users specify out features as lists,
        # assume dimensions to be all zero

        # bookkeeping
        self.out_features = out_features
        self.levels = ["onefour", "nonbonded"]

        # get output features
        mid_features = [x for x in config if isinstance(x, int)][-1]

        # set up networks
        for level in self.levels:

            # set up individual sequential networks
            setattr(
                self,
                "sequential_%s" % level,
                esp.nn.sequential._Sequential(
                    in_features=2 * in_features,
                    config=config,
                    layer=torch.nn.Linear,
                ),
            )

            for feature, dimension in self.out_features.items():
                setattr(
                    self,
                    "f_out_%s_to_%s" % (level, feature),
                    torch.nn.Linear(
                        mid_features,
                        dimension,
                    ),
                )

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph,
            input graph.
        """

        # copy
        g.multi_update_all(
            {
                "n1_as_%s_in_%s"
                % (relationship_idx, big_idx): (
                    dgl.function.copy_u("h", "m%s" % relationship_idx),
                    dgl.function.mean(
                        "m%s" % relationship_idx, "h%s" % relationship_idx
                    ),
                )
                for big_idx in self.levels
                for relationship_idx in range(2)
            },
            cross_reducer="sum",
        )

        for big_idx in self.levels:

            g.apply_nodes(
                func=lambda nodes: {
                    feature: getattr(
                        self, "f_out_%s_to_%s" % (big_idx, feature)
                    )(
                        torch.sum(
                            torch.stack(
                                [
                                    getattr(self, "sequential_%s" % big_idx)(
                                        g=None,
                                        x=torch.cat(
                                            [
                                                nodes.data["h0"],
                                                nodes.data["h1"],
                                            ],
                                            dim=1,
                                        ),
                                    ),
                                    getattr(self, "sequential_%s" % big_idx)(
                                        g=None,
                                        x=torch.cat(
                                            [
                                                nodes.data["h1"],
                                                nodes.data["h0"],
                                            ],
                                            dim=1,
                                        ),
                                    ),
                                ],
                                dim=0,
                            ),
                            dim=0,
                        )
                    )
                    for feature in self.out_features.keys()
                },
                ntype=big_idx,
            )

        return g


class ExpCoefficients(torch.nn.Module):
    def forward(self, g):
        import math

        g.nodes["n2"].data["coefficients"] = (
            g.nodes["n2"].data["log_coefficients"].exp()
        )
        g.nodes["n3"].data["coefficients"] = (
            g.nodes["n3"].data["log_coefficients"].exp()
        )
        return g


class LinearMixtureToOriginal(torch.nn.Module):
    def forward(self, g):
        import math

        (
            g.nodes["n2"].data["k"],
            g.nodes["n2"].data["eq"],
        ) = esp.mm.functional.linear_mixture_to_original(
            g.nodes["n2"].data["coefficients"][:, 0][:, None],
            g.nodes["n2"].data["coefficients"][:, 1][:, None],
            1.5,
            6.0,
        )

        (
            g.nodes["n3"].data["k"],
            g.nodes["n3"].data["eq"],
        ) = esp.mm.functional.linear_mixture_to_original(
            g.nodes["n3"].data["coefficients"][:, 0][:, None],
            g.nodes["n3"].data["coefficients"][:, 1][:, None],
            0.0,
            math.pi,
        )

        g.nodes["n3"].data.pop("coefficients")
        g.nodes["n2"].data.pop("coefficients")
        return g




class JanossyPoolingOOP(torch.nn.Module):
    """Janossy pooling (arXiv:1811.01900) to average node representation
    for oop bending.
    """

    def __init__(
        self,
        config,
        in_features,
        out_features={
            "k": 2,
        },
        out_features_dimensions=-1,
    ):
        super(JanossyPoolingOOP, self).__init__()

        # if users specify out features as lists,
        # assume dimensions to be all zero

        # bookkeeping
        self.out_features = out_features
        self.levels = ["n4_oop"]

        # get output features
        mid_features = [x for x in config if isinstance(x, int)][-1]

        # set up networks
        for level in self.levels:

            # set up individual sequential networks
            setattr(
                self,
                "sequential_%s" % level,
                esp.nn.sequential._Sequential(
                    in_features=4 * in_features,
                    config=config,
                    layer=torch.nn.Linear,
                ),
            )

            for feature, dimension in self.out_features.items():
                setattr(
                    self,
                    "f_out_%s_to_%s" % (level, feature),
                    torch.nn.Linear(
                        mid_features,
                        dimension,
                    ),
                )

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph,
            input graph.
        """
        import dgl

        # copy
        g.multi_update_all(
            {
                "n1_as_%s_in_%s"
                % (relationship_idx, big_idx): (
                    dgl.function.copy_u("h", "m%s" % relationship_idx),
                    dgl.function.mean(
                        "m%s" % relationship_idx, "h%s" % relationship_idx
                    ),
                )
                for big_idx in self.levels
                for relationship_idx in range(4)
            },
            cross_reducer="sum",
        )

        if g.number_of_nodes("n4_improper") == 0:
            return g

        # pool
        #   sum over three cyclic permutations of "h0", "h2", "h3", assuming "h1" is the central atom in the improper
        #   following the smirnoff trefoil convention [(0, 1, 2, 3), (2, 1, 3, 0), (3, 1, 0, 2)]
        #   https://github.com/openff.toolkit/openff.toolkit/blob/166c9864de3455244bd80b2c24656bd7dda3ae2d/openff.toolkit/typing/engines/smirnoff/parameters.py#L3326-L3360

        # TODO to check
        ## Set different permutations based on which definition of impropers
        ##  are being used
        permuts = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
        stack_permuts = lambda nodes, p: torch.cat(
            [nodes.data[f"h{i}"] for i in p], dim=1
        )

        for big_idx in self.levels:
            inner_net = getattr(self, f"sequential_{big_idx}")

            g.apply_nodes(
                func=lambda nodes: {
                    feature: getattr(self, f"f_out_{big_idx}_to_{feature}")(
                        torch.sum(
                            torch.stack(
                                [
                                    inner_net(
                                        g=None, x=stack_permuts(nodes, p)
                                    )
                                    for p in permuts
                                ],
                                dim=0,
                            ),
                            dim=0,
                        )
                    )
                    for feature in self.out_features.keys()
                },
                ntype=big_idx,
            )

        return g
