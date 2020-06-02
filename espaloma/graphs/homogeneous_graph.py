# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl
import rdkit
import openforcefield
import torch

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HomogeneousGraph(esp.Graph, dgl.DGLGraph):
    r""" Homogeneous graph that contains no more than connectivity and
    atom attributes.

    Parameters
    ----------
    mol : a `rdkit.Chem.Molecule` or `openeye.GraphMol` object

    """

    def __init__(self, mol=None):
        super(HomogeneousGraph, self).__init__()
        self.set_stage(type='homogeneous')

        if mol is not None:
            if isinstance(mol, esp.MoleculeGraph):
                mol.to_homogeneous_graph(self)
            else:
                mol = esp.MoleculeGraph(mol)

                mol.to_homogeneous_graph(self)

    def loss(self, level, *args, **kwargs):
        """ Loss function between attributes in the graph.

        """

        if level == 'node_classification':

            return self._loss_node_classification(
                    *args, **kwargs)

        else:
            raise NotImplementedError

    def legacy_typing(self):
        assert self.stage['legacy_typed'] == True
        return self.ndata['legacy_type']

    def nn_typing(self):
        assert self.stage['neuralized'] == True
        return self.ndata['nn_type']

