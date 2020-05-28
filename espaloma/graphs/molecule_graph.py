# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
from openforcefield.topology import Molecule

# =============================================================================
# MODULE CLASSES
# =============================================================================
class MoleculeGraph(esp.Graph, Molecule):
    """ Base class of molecule graph.


    """

    def __init__(self, mol=None):
        super(MoleculeGraph, self).__init__()
        set_stage(type, 'molecule')

        if mol is not None: # support reading mol this way
            if isinstance(mol, rdkit.Chem.rdchem.Mol):
                self.from_rdkit(mol)

            elif 'openeye' in str(type(mol)):
                self.from_opeyeye(mol)

            else:
                raise RuntimeError(
                    "Input molecule could only be"
                    " one of RDKit, OpenEye, or OpenForceField.")

    def to_homogeneous_graph(self, g=None):
        """ Add nodes and edges to a graph.
        """
        # initialize empty graph if there isn't one
        if g is None:
            g = esp.HomogeneousGraph()

        # TODO:
        # Use openforcefield-generic grammar
        esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(
            g, self.to_rdkit())
