import dgl
import numpy as np
import torch

from ..graph import Graph
from .offmol_indices import out_of_plane_indices


def generate_oops(g: Graph):
    """
    Method to regenerate the improper nodes according to the specified
    method of permuting the impropers. Modifies the esp.Graph's heterograph
    in place and returns the new heterograph.
    NOTE: This will clear the data on all n4_improper nodes, including
    previously generated improper from JanossyPoolingImproper.
    """

    ## First get rid of the old nodes/edges
    hg = g.heterograph

    # Extract representation data that closely resembles the original data_dict
    representation_data_dict = {}
    for edge_type in hg.canonical_etypes:
        src_type, edge_name, dst_type = edge_type
        src, dst = hg.edges(etype=edge_type)
        representation_data_dict[edge_type] = (src, dst)
    
    idxs = out_of_plane_indices(g.mol)

    ## New edges b/n oop permuts and n1 nodes
    permut_ids = np.arange(idxs.shape[0])
    for i in range(4):
        n1_ids = idxs[:, i]

        # edge from improper node to n1 node
        outgoing_etype = ("n4_oop", f"n4_oop_has_{i}_n1", "n1")
        
        representation_data_dict[outgoing_etype] = (permut_ids, n1_ids)

        # edge from n1 to improper
        incoming_etype = ("n1", f"n1_as_{i}_in_n4_oop", "n4_oop")
        representation_data_dict[incoming_etype] = (n1_ids, permut_ids)


    ## New edges b/n improper permuts and the graph (for global pooling)
    # edge from improper node to graph
    
    outgoing_etype = ("n4_oop", f"n4_oop_in_g", "g")
    representation_data_dict[outgoing_etype] = (permut_ids, np.zeros_like(permut_ids))

    # edge from graph to improper nodes
    incoming_etype = ("g", "g_has_n4_oop", "n4_oop")
    representation_data_dict[incoming_etype] = (np.zeros_like(permut_ids), permut_ids)
    


    new_hg = dgl.heterograph(representation_data_dict)
    for node_type in hg.ntypes:
        ndata = hg.nodes[node_type].data
        for k in ndata.keys():
            new_hg.nodes[node_type].data[k] = ndata[k]

    new_hg.nodes["n4_oop"].data["idxs"] = torch.tensor(idxs)

    g.heterograph = new_hg

    return g  # hg
