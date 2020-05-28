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
                self.from_rdkit(mol)

            elif isinstance(mol, openforcefield.topology.molecule.Molecule):
                self.from_rdkit(mol.to_rdkit())

            elif "oe" in str(type(mol)):  # we don't want to depend on OE
                self.from_openeye(mol)

            else:
                raise RuntimeError(
                    "Input molecule could only be"
                    " one of RDKit, OpenEye, or OpenForceField."
                )


    @property
    def stage(self):
        return {'type': 'homogeneous'}

    def from_rdkit(self, mol):
        # TODO: raise error if this is called after a class has been
        # initialized
        esp.graphs.utils.read_homogeneous_graph.from_rdkit_mol(self, mol)

    def from_openeye(self, mol):
        esp.graphs.utils.read_homogeneous_graph.from_openeye_mol(self, mol)
