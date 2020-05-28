# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl
import rdkit
import openforcefield

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

        if mol is not None:
            if isinstance(mol, rdkit.Chem.rdchem.Mol):
                self._from_rdkit(mol)

            elif isinstance(mol, openforcefield.topology.molecule.Molecule):
                self._from_rdkit(mol.to_rdkit())

            elif "oe" in str(type(mol)):  # we don't want to depend on OE
                self._from_openeye(mol)

            else:
                raise RuntimeError(
                    "Input molecule could only be"
                    " one of RDKit, OpenEye, or OpenForceField."
                )


    def _from_rdkit(self, mol):
        """ API to read RDKit mol.

        Parameters
        ----------
        mol : `rdkit.Chem.rdchem.Mol` object

        """
        # TODO: raise error if this is called after a class has been
        # initialized
        esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(self, mol)

    def _from_openeye(self, mol):
        """ API to read OpenEye mol.

        Parameters
        ----------
        mol : `openeye.oechem.GraphMol` object
        """
        esp.graphs.utils.read_homogeneous_graph.from_openeye_mol(self, mol)

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
   
    def gn_typing(self):
        pass

    def _loss_node_classification(
            self, 
            loss_fn=torch.nn.functional.cross_entropy):

        raise NotImplementedError
                
