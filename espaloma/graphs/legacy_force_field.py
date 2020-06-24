# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import torch
import abc
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
import rdkit

# =============================================================================
# CONSTANTS
# =============================================================================
REDUNDANT_TYPES = {
    "cd": "cc",
    "cf": "ce",
    "cq": "cp",
    "pd": "pc",
    "pf": "pe",
    "nd": "nc",
}


# =============================================================================
# MODULE CLASSES
# =============================================================================
class LegacyForceField:
    """ Class to hold legacy forcefield for typing and parameter assignment.

    Parameters
    ----------
    forcefield : string
        name and version of the forcefield.

    """

    def __init__(self, forcefield="gaff-1.81"):
        self.forcefield = forcefield
        self._prepare_forcefield()

    @staticmethod
    def _convert_to_off(mol):
        import openforcefield

        if isinstance(mol, esp.Graph):
            return mol.mol

        elif isinstance(mol, openforcefield.topology.molecule.Molecule):
            return mol
        elif isinstance(mol, rdkit.Chem.rdchem.Mol):
            return Molecule.from_rdkit(mol)
        elif "openeye" in str(type(mol)):  # because we don't want to depend on OE
            return Molecule.from_openeye(mol)

    def _prepare_forcefield(self):
        if "gaff" in self.forcefield:
            self._prepare_gaff()

        else:
            raise NotImplementedError

    def _prepare_gaff(self):
        import os
        import openmmforcefields
        import xml.etree.ElementTree as ET

        # get the openforcefields path
        openmmforcefields_path = os.path.dirname(openmmforcefields.__file__)

        # get the xml path
        ffxml_path = (
            openmmforcefields_path
            + "/ffxml/amber/gaff/ffxml/"
            + self.forcefield
            + ".xml"
        )

        # parse xml
        tree = ET.parse(ffxml_path)
        root = tree.getroot()
        nonbonded = root.getchildren()[-1]
        atom_types = [atom.get("type") for atom in nonbonded.findall("Atom")]

        # remove redundant types
        [atom_types.remove(bad_type) for bad_type in REDUNDANT_TYPES.keys()]

        # compose the translation dictionaries
        str_2_idx = dict(zip(atom_types, range(len(atom_types))))
        idx_2_str = dict(zip(range(len(atom_types)), atom_types))

        # provide mapping for redundant types
        for bad_type, good_type in REDUNDANT_TYPES.items():
            str_2_idx[bad_type] = str_2_idx[good_type]

        # make translation dictionaries attributes of self
        self._str_2_idx = str_2_idx
        self._idx_2_str = idx_2_str

    def _type_gaff(self, mol, g=None):
        """ Type a molecular graph using gaff force fields.

        """
        # assert the forcefield is indeed of gaff family
        assert "gaff" in self.forcefield

        # make sure mol is in OpenForceField format `
        mol = self._convert_to_off(mol)

        # import template generator
        from openmmforcefields.generators import GAFFTemplateGenerator

        gaff = GAFFTemplateGenerator(molecules=mol, forcefield=self.forcefield)

        # create temporary directory for running antechamber
        import tempfile
        import os
        import shutil

        tempdir = tempfile.mkdtemp()
        prefix = "molecule"
        input_sdf_filename = os.path.join(tempdir, prefix + ".sdf")
        gaff_mol2_filename = os.path.join(tempdir, prefix + ".gaff.mol2")
        frcmod_filename = os.path.join(tempdir, prefix + ".frcmod")

        # write sdf for input
        mol.to_file(input_sdf_filename, file_format="sdf")

        # run antechamber
        gaff._run_antechamber(
            molecule_filename=input_sdf_filename,
            input_format="mdl",
            gaff_mol2_filename=gaff_mol2_filename,
            frcmod_filename=frcmod_filename,
        )

        gaff._read_gaff_atom_types_from_mol2(gaff_mol2_filename, mol)
        gaff_types = [atom.gaff_type for atom in mol.atoms]
        shutil.rmtree(tempdir)

        # put types into graph object
        if g is None:
            g = esp.Graph(mol)

        g.nodes['n1'].data["legacy_typing"] = torch.tensor(
            [self._str_2_idx[atom] for atom in gaff_types]
        )

        return g

    def typing(self, mol, g=None):
        """ Type a molecular graph.

        """
        if "gaff" in self.forcefield:
            return self._type_gaff(mol, g)

        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.typing(*args, **kwargs)
