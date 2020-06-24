""" Build heterogeneous graph from homogeneous ones.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import numpy as np

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def relationship_indices_from_adjacency_matrix(a, max_size=4):
    r""" Read the relatinoship indices from adjacency matrix.

    Parameters
    ----------
    a : torch.sparse.FloatTensor
        adjacency matrix.

    Returns
    -------
    idxs : dictionary
        that contains the indices of subgraphs of various size.
    """

    # make sure max size is larger than 2
    assert isinstance(max_size, int)
    assert max_size >= 2

    idxs = {}

    # get the indices of n2
    idxs["n2"] = a._indices().t().detach()  # just in case

    # loop through the levels
    for level in range(3, max_size + 1):
        # get the indices that is the basis of the level
        base_idxs = idxs["n%s" % (level - 1)]

        # enumerate all the possible pairs at base level
        base_pairs = torch.cat(
            [
                base_idxs[None, :, :].repeat(base_idxs.shape[0], 1, 1),
                base_idxs[:, None, :].repeat(1, base_idxs.shape[0], 1),
            ],
            dim=-1,
        ).reshape(-1, 2 * (level - 1))

        mask = 1.0
        # filter to get the ones that share some indices
        for idx_pos in range(level - 2):
            mask *= torch.eq(
                base_pairs[:, idx_pos + 1], base_pairs[:, idx_pos + level - 1]
            )

        mask *= 1 - 1 * torch.eq(base_pairs[:, 0], base_pairs[:, -1])

        mask = mask > 0.0

        # filter the enumeration to be output
        idxs_level = torch.cat(
            [base_pairs[mask][:, : (level - 1)], base_pairs[mask][:, -1][:, None]],
            dim=-1,
        )

        idxs["n%s" % level] = idxs_level

    return idxs


def from_homogeneous(g):
    r""" Build heterogeneous graph from homogeneous ones.


    Note
    ----
    For now we name single node, two-, three, and four-,
    hypernodes as `n1`, `n2`, `n3`, and `n4`. These correspond
    to atom, bond, angle, and torsion nodes in chemical graphs.


    Parameters
    ----------
    g : `espaloma.HomogeneousGraph` object
        the homogeneous graph to be translated.

    Returns
    -------
    hg : `espaloma.HeterogeneousGraph` object
        the resulting heterogeneous graph.

    """

    # initialize empty dictionary
    hg = {}

    # get all the indices
    idxs = relationship_indices_from_adjacency_matrix(g.adjacency_matrix())

    # make them all numpy
    idxs = {key: value.numpy() for key, value in idxs.items()}

    # also include n1
    idxs["n1"] = np.arange(g.number_of_nodes())[:, None]

    # NOTE:
    # here we only define the neighboring relationship
    # on atom level
    for idx in range(1, 5):
        hg[("n%s" % idx, "n%s_neighbors_n%s" % (idx, idx), "n%s" % idx)] = idxs["n2"]

    # build a mapping between indices and the ordering
    idxs_to_ordering = {}

    for term in ["n1", "n2", "n3", "n4"]:
        idxs_to_ordering[term] = {
            tuple(subgraph_idxs): ordering
            for (ordering, subgraph_idxs) in enumerate(list(idxs[term]))
        }

    # NOTE:
    # here we define all the possible
    # 'has' and 'in' relationships.
    # TODO:
    # we'll test later to see if this adds too much overhead
    for small_idx in range(1, 5):
        for big_idx in range(small_idx + 1, 5):
            for pos_idx in range(big_idx - small_idx):
                hg[
                    (
                        "n%s" % small_idx,
                        "n%s_as_%s_in_n%s" % (small_idx, pos_idx, big_idx),
                        "n%s" % big_idx,
                    )
                ] = np.stack(
                    [
                        np.array(
                            [
                                idxs_to_ordering["n%s" % small_idx][tuple(x)]
                                for x in idxs["n%s" % big_idx][
                                    :, pos_idx : pos_idx + small_idx
                                ]
                            ]
                        ),
                        np.arange(idxs["n%s" % big_idx].shape[0]),
                    ],
                    axis=1,
                )

                hg[
                    (
                        "n%s" % big_idx,
                        "n%s_has_%s_n%s" % (big_idx, pos_idx, small_idx),
                        "n%s" % small_idx,
                    )
                ] = np.stack(
                    [
                        np.arange(idxs["n%s" % big_idx].shape[0]),
                        np.array(
                            [
                                idxs_to_ordering["n%s" % small_idx][tuple(x)]
                                for x in idxs["n%s" % big_idx][
                                    :, pos_idx : pos_idx + small_idx
                                ]
                            ]
                        ),
                    ],
                    axis=1,
                )

    hg = dgl.heterograph({key: list(value) for key, value in hg.items()})

    hg.nodes["n1"].data["h0"] = g.ndata["h0"]

    return hg
