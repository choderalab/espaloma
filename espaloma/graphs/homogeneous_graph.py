# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import dgl

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
            if 'rdkit' in str(type(mol)):
                self.from_rdkit(mol)

            elif 'oe' in str(type(mol)):
                self.from_openeye(mol)

            elif 'openforcefield' in str(type(mol)):
                self.from_rdkit(mol.to_rdkit())


    @property
    def _stage(self):
        return 'homogeneous'

    def from_rdkit(self, mol):
        esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(self, mol)

    def from_openeye(self, mol):
        esp.graphs.utils.read_homogeneous_graph.from_openeye_mol(self, mol)

