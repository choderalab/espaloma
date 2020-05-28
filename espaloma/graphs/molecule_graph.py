# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
from openforcefield.topology import Molecule
import rdkit

# =============================================================================
# MODULE CLASSES
# =============================================================================
class MoleculeGraph(esp.Graph):
    """ Base class of molecule graph.


    """

    def __init__(self, mol=None):
        super(MoleculeGraph, self).__init__()
        self.set_stage(type='molecule')

        # in such case there won't be a molecule
        self._g = None

        if mol is not None: # support reading mol this way
            if isinstance(mol, Molecule):
                self._g = mol
            
            elif isinstance(mol, rdkit.Chem.rdchem.Mol):
                self._g = Molecule.from_rdkit(mol)

            elif 'openeye' in str(type(mol)):
                self._g = Molecule.from_opeyeye(mol)

            else:
                raise RuntimeError(
                    "Input molecule could only be"
                    " one of RDKit, OpenEye, or OpenForceField, got %s" %\
                            type(mol))

    def to_homogeneous_graph(self, g=None):
        """ Add nodes and edges to a graph.
        """
        # initialize empty graph if there isn't one
        if g is None:
            g = esp.HomogeneousGraph()

        # TODO:
        # Use openforcefield-generic grammar
        esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(
            g, self._g.to_rdkit())
