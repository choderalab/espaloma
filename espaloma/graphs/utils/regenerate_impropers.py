import dgl
import numpy as np
import torch

from ..graph import Graph
from .offmol_indices import improper_torsion_indices


def regenerate_impropers(g: Graph, improper_def="smirnoff"):
    """
    Method to regenerate the improper nodes according to the specified
    method of permuting the impropers. Modifies the esp.Graph's heterograph
    in place and returns the new heterograph.
    NOTE: This will clear the data on all n4_improper nodes, including
    previously generated improper from JanossyPoolingImproper.
    """

    ## First get rid of the old nodes/edges
    hg = g.heterograph
    hg = dgl.remove_nodes(hg, hg.nodes("n4_improper"), "n4_improper")

    ## Generate new improper torsion permutations
    idxs = improper_torsion_indices(g.mol, improper_def)
    if len(idxs) == 0:
        return g

    ## Add new nodes of type n4_improper (one for each permut)
    hg = dgl.add_nodes(hg, idxs.shape[0], ntype="n4_improper")

    ## New edges b/n improper permuts and n1 nodes
    permut_ids = np.arange(idxs.shape[0])
    for i in range(4):
        n1_ids = idxs[:, i]

        # edge from improper node to n1 node
        outgoing_etype = ("n4_improper", f"n4_improper_has_{i}_n1", "n1")
        hg = dgl.add_edges(hg, permut_ids, n1_ids, etype=outgoing_etype)

        # edge from n1 to improper
        incoming_etype = ("n1", f"n1_as_{i}_in_n4_improper", "n4_improper")
        hg = dgl.add_edges(hg, n1_ids, permut_ids, etype=incoming_etype)

    ## New edges b/n improper permuts and the graph (for global pooling)
    # edge from improper node to graph
    outgoing_etype = ("n4_improper", f"n4_improper_in_g", "g")
    hg = dgl.add_edges(
        hg, permut_ids, np.zeros_like(permut_ids), etype=outgoing_etype
    )

    # edge from graph to improper nodes
    incoming_etype = ("g", "g_has_n4_improper", "n4_improper")
    hg = dgl.add_edges(
        hg, np.zeros_like(permut_ids), permut_ids, etype=incoming_etype
    )

    hg.nodes["n4_improper"].data["idxs"] = torch.tensor(idxs)
    
    g.heterograph = hg

    return g  # hg
