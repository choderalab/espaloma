""" Build heterogeneous graph from homogeneous ones.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import numpy as np
import torch
from espaloma.graphs.utils import offmol_indices
from openforcefield.topology import Molecule
from typing import Dict

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def duplicate_index_ordering(indices: np.ndarray) -> np.ndarray:
    """For every (a,b,c,d) add a (d,c,b,a)

    >>> indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    >>> duplicate_index_ordering(indices)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4],
           [3, 2, 1, 0],
           [4, 3, 2, 1]])
    """
    return np.vstack([indices, indices[:, ::-1]])


def relationship_indices_from_offmol(offmol: Molecule) -> Dict[str, torch.Tensor]:
    """Construct a dictionary that maps node names (like "n2") to torch tensors of indices

    Notes
    -----
    * introduces 2x redundant indices (including (d,c,b,a) for every (a,b,c,d)) for compatibility with later processing
    """
    idxs = dict()
    idxs["n2"] = offmol_indices.bond_indices(offmol)
    idxs["n3"] = offmol_indices.angle_indices(offmol)
    idxs["n4"] = offmol_indices.proper_torsion_indices(offmol)

    # TODO (pending discussion with YW) : list n4_improper inside n4 ?
    idxs["n4_improper"] = offmol_indices.improper_torsion_indices(offmol)

    # TODO: enumerate indices for coupling-term nodes also
    # TODO: big refactor of term names from "n4" to "proper_torsion", "improper_torsion", "angle_angle_coupling", etc.

    for key in idxs:
        idxs[key] = torch.from_numpy(duplicate_index_ordering(idxs[key]))

    return idxs


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
            [
                base_pairs[mask][:, : (level - 1)],
                base_pairs[mask][:, -1][:, None],
            ],
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

    # get adjacency matrix
    a = g.adjacency_matrix()

    # get all the indices
    idxs = relationship_indices_from_adjacency_matrix(a)

    # make them all numpy
    idxs = {key: value.numpy() for key, value in idxs.items()}

    # also include n1
    idxs["n1"] = np.arange(g.number_of_nodes())[:, None]

    # =========================
    # neighboring relationships
    # =========================
    # NOTE:
    # here we only define the neighboring relationship
    # on atom level
    hg[("n1", "n1_neighbors_n1", "n1")] = idxs["n2"]

    # build a mapping between indices and the ordering
    idxs_to_ordering = {}

    for term in ["n1", "n2", "n3", "n4"]:
        idxs_to_ordering[term] = {
            tuple(subgraph_idxs): ordering
            for (ordering, subgraph_idxs) in enumerate(list(idxs[term]))
        }

    # ===============================================
    # relationships between nodes of different levels
    # ===============================================
    # NOTE:
    # here we define all the possible
    # 'has' and 'in' relationships.
    # TODO:
    # we'll test later to see if this adds too much overhead
    for small_idx in range(1, 5):
        for big_idx in range(small_idx + 1, 5):
            for pos_idx in range(big_idx - small_idx + 1):
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

    # ======================================
    # nonbonded terms
    # ======================================
    # NOTE: everything is counted twice here
    # nonbonded is where
    # $A = AA = AAA = AAAA = 0$

    # make dense
    a_ = a.to_dense().detach().numpy()

    idxs["nonbonded"] = np.stack(
        np.where(
            np.equal(a_ + a_ @ a_ + a_ @ a_ @ a_ + a_ @ a_ @ a_ @ a_, 0.0)
        ),
        axis=-1,
    )

    # onefour is the two ends of torsion
    idxs["onefour"] = np.stack([idxs["n4"][:, 0], idxs["n4"][:, 3],], axis=1)

    # membership
    for term in ["nonbonded", "onefour"]:
        for pos_idx in [0, 1]:
            hg[(term, "%s_has_%s_n1" % (term, pos_idx), "n1")] = np.stack(
                [np.arange(idxs[term].shape[0]), idxs[term][:, pos_idx]],
                axis=-1,
            )

            hg[("n1", "n1_as_%s_in_%s" % (pos_idx, term), term)] = np.stack(
                [idxs[term][:, pos_idx], np.arange(idxs[term].shape[0]),],
                axis=-1,
            )

    # ======================================
    # relationships between nodes and graphs
    # ======================================
    for term in ["n1", "n2", "n3", "n4", "nonbonded", "onefour"]:
        hg[(term, "%s_in_g" % term, "g",)] = np.stack(
            [np.arange(len(idxs[term])), np.zeros(len(idxs[term]))], axis=1,
        )

        hg[("g", "g_has_%s" % term, term)] = np.stack(
            [np.zeros(len(idxs[term])), np.arange(len(idxs[term])),], axis=1,
        )

    hg = dgl.heterograph({key: list(value) for key, value in hg.items()})

    hg.nodes["n1"].data["h0"] = g.ndata["h0"]

    # include indices in the nodes themselves
    for term in ["n1", "n2", "n3", "n4"]:
        hg.nodes[term].data["idxs"] = torch.tensor(idxs[term])

    return hg
