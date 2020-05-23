# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
import abc
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HomogeneousGraph(espaloma.Graph, dgl.DGLGraph):
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

            if 'oe' in str(type(mol)):
                self.from_openeye(mol)


    @property
    def _stage(self):
        return 'homogeneous'

    def from_rdkit(self, mol):
        from utils.read_homogeneous_graph import from_rdkit_mol
        from_rdkit_mol(self, mol)

    def from_openeye(self, mol):
        from utils.read_homogeneous_graph import from_oemol
        from_openeye_mol(self, mol)

