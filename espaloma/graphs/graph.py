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

    def __init__(self, mol=None, homograph=None, heterograph=None):
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
                homograph, mol
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

    @classmethod
    def load(cls, path):
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
    def get_heterograph_from_graph_and_mol(graph, mol):
        import dgl
        assert isinstance(
            graph, dgl.DGLGraph
        ), "graph can only be dgl Graph object."

        heterograph = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous_and_mol(
            graph, mol
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
