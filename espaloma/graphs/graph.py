# =============================================================================
# IMPORTS
# =============================================================================
import abc
import openff.toolkit

import espaloma as esp


# =============================================================================
# MODULE CLASSES
# =============================================================================
class BaseGraph(abc.ABC):
    """ Base class of graph. """

    def __init__(self):
        super(BaseGraph, self).__init__()


class Graph(BaseGraph):
    """ A unified graph object that support translation to and from
    message-passing graphs and MM factor graph.

    Methods
    -------
    save(path)
        Save graph to file.

    load(path)
        Load a graph from path.

    Note
    ----
    This object provides access to popular attributes of homograph and
    heterograph.

    This object also provides access to `ndata` and `edata` from the heterograph.

    Examples
    --------
    >>> g0 = esp.Graph("C")
    >>> g1 = esp.Graph(Molecule.from_smiles("C"))
    >>> assert g0 == g1

    """

    def __init__(self, mol=None, homograph=None, heterograph=None,
        improper_def='smirnoff'):
        # TODO : more pythonic way allow multiple constructors:
        #   Graph.from_smiles(...), Graph.from_mol(...), Graph.from_homograph(...), ...
        #   rather than Graph(mol=None, homograph=None, ...)

        # input molecule
        if isinstance(mol, str):
            from openff.toolkit.topology import Molecule

            mol = Molecule.from_smiles(mol, allow_undefined_stereo=True)

        if mol is not None and homograph is None and heterograph is None:
            homograph = self.get_homograph_from_mol(mol)

        if homograph is not None and heterograph is None:
            heterograph = self.get_heterograph_from_graph_and_mol(
                homograph, mol, improper_def
            )

        self.mol = mol
        self.homograph = homograph
        self.heterograph = heterograph

    def save(self, path):
        import os
        import json
        import dgl
        os.mkdir(path)
        dgl.save_graphs(path + "/homograph.bin", [self.homograph])
        dgl.save_graphs(path + "/heterograph.bin", [self.heterograph])
        with open(path + "/mol.json", "w") as f_handle:
            json.dump(self.mol.to_json(), f_handle)

    def regenerate_impropers(self, improper_def='smirnoff'):
        """
        Method to regenerate the improper nodes according to the specified
        method of permuting the impropers.
        NOTE: This will clear the data on all n4_improper nodes, including
        previously generated improper from JanossyPoolingImproper.
        """

        import dgl
        import numpy as np
        import torch

        from .utils.offmol_indices import improper_torsion_indices

        ## First get rid of the old nodes/edges
        g = self.heterograph
        g = dgl.remove_nodes(g, g.nodes('n4_improper'), 'n4_improper')

        ## Generate new improper torsion permutations
        idxs = improper_torsion_indices(self.mol, improper_def)
        if len(idxs) == 0:
            return

        ## Add new nodes of type n4_improper (one for each permut)
        g = dgl.add_nodes(g, idxs.shape[0], ntype='n4_improper')

        ## New edges b/n improper permuts and n1 nodes
        permut_ids = np.arange(idxs.shape[0])
        for i in range(4):
            n1_ids = idxs[:,i]

            # edge from improper node to n1 node
            outgoing_etype = ('n4_improper', f'n4_improper_has_{i}_n1', 'n1')
            g = dgl.add_edges(g, permut_ids, n1_ids, etype=outgoing_etype)

            # edge from n1 to improper
            incoming_etype = ('n1', f'n1_as_{i}_in_n4_improper', 'n4_improper')
            g = dgl.add_edges(g, n1_ids, permut_ids, etype=incoming_etype)

        self.heterograph = g

    @classmethod
    def load(cls, path, improper_def=None):
        import json
        import dgl

        homograph = dgl.load_graphs(path + "/homograph.bin")[0][0]
        heterograph = dgl.load_graphs(path + "/heterograph.bin")[0][0]

        with open(path + "/mol.json", "r") as f_handle:
            mol = json.load(f_handle)
        from openff.toolkit.topology import Molecule

        try:
            mol = Molecule.from_json(mol)
        except:
            mol = Molecule.from_dict(mol)

        g = cls(mol=mol, homograph=homograph, heterograph=heterograph)
        if improper_def is not None:
            g.regenerate_impropers(improper_def)
        return g

    @staticmethod
    def get_homograph_from_mol(mol):
        assert isinstance(
            mol, openff.toolkit.topology.Molecule
        ), "mol can only be OFF Molecule object."

        # TODO:
        # rewrite this using OFF-generic grammar
        # graph = esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(
        #     mol.to_rdkit()
        # )

        graph = (
            esp.graphs.utils.read_homogeneous_graph.from_openff_toolkit_mol(
                mol
            )
        )
        return graph

    @staticmethod
    def get_heterograph_from_graph_and_mol(graph, mol, improper_def='smirnoff'):
        import dgl
        assert isinstance(
            graph, dgl.DGLGraph
        ), "graph can only be dgl Graph object."

        heterograph = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous_and_mol(
            graph, mol, improper_def
        )

        return heterograph

    #
    # @property
    # def mol(self):
    #     return self._mol
    #
    # @property
    # def homograph(self):
    #     return self._homograph
    #
    # @property
    # def heterograph(self):
    #     return self._heterograph

    @property
    def ndata(self):
        return self.homograph.ndata

    @property
    def edata(self):
        return self.homograph.edata

    @property
    def nodes(self):
        return self.heterograph.nodes
