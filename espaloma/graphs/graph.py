# =============================================================================
# IMPORTS
# =============================================================================
import abc

import dgl
import openforcefield

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

    Note
    ----
    This object provides access to popular attributes of homograph and
    heterograph.

    """

    def __init__(self, mol=None, homograph=None, heterograph=None):
        # input molecule
        if isinstance(mol, str):
            from openforcefield.topology import Molecule

            mol = Molecule.from_smiles(mol, allow_undefined_stereo=True)

        if mol is not None and homograph is None and heterograph is None:
            homograph = self.get_homograph_from_mol(mol)

        if homograph is not None and heterograph is None:
            heterograph = self.get_heterograph_from_graph(homograph)

        self.mol = mol
        self.homograph = homograph
        self.heterograph = heterograph

    @staticmethod
    def get_homograph_from_mol(mol):
        assert isinstance(
            mol, openforcefield.topology.Molecule
        ), "mol can only be OFF Molecule object."

        # TODO:
        # rewrite this using OFF-generic grammar
        # graph = esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(
        #     mol.to_rdkit()
        # )

        graph = esp.graphs.utils.read_homogeneous_graph.from_openforcefield_mol(
            mol
        )
        return graph

    @staticmethod
    def get_heterograph_from_graph(graph):
        assert isinstance(
            graph, dgl.DGLGraph
        ), "graph can only be dgl Graph object."

        heterograph = esp.graphs.utils.read_heterogeneous_graph.from_homogeneous(
            graph
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

    def save(self, path):
        import pickle

        with open(path, "wb") as f_handle:
            pickle.dump([self.mol, self.homograph, self.heterograph], f_handle)

    def load(self, path):
        import pickle

        with open(path, "rb") as f_handle:
            (self.mol, self.homograph, self.heterograph) = pickle.load(
                f_handle
            )
