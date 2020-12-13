# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
import torch
from openforcefield.topology import Molecule
import espaloma as esp

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
        elif "openeye" in str(
            type(mol)
        ):  # because we don't want to depend on OE
            return Molecule.from_openeye(mol)

    def _prepare_forcefield(self):

        if "gaff" in self.forcefield:
            self._prepare_gaff()

        elif "smirnoff" in self.forcefield:
            # do nothing for now
            self._prepare_smirnoff()

        elif "openff" in self.forcefield:
            self._prepare_openff()

        else:
            raise NotImplementedError

    def _prepare_openff(self):

        from openforcefield.typing.engines.smirnoff import ForceField

        self.FF = ForceField("%s.offxml" % self.forcefield)

    def _prepare_smirnoff(self):

        from openforcefield.typing.engines.smirnoff import ForceField

        self.FF = ForceField("test_forcefields/%s.offxml" % self.forcefield)

    def _prepare_gaff(self):
        import os
        import xml.etree.ElementTree as ET

        import openmmforcefields

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

    def _type_gaff(self, g):
        """ Type a molecular graph using gaff force fields.

        """
        # assert the forcefield is indeed of gaff family
        assert "gaff" in self.forcefield

        # make sure mol is in OpenForceField format `
        mol = g.mol

        # import template generator
        from openmmforcefields.generators import GAFFTemplateGenerator

        gaff = GAFFTemplateGenerator(molecules=mol, forcefield=self.forcefield)

        # create temporary directory for running antechamber
        import os
        import shutil
        import tempfile

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

        g.nodes["n1"].data["legacy_typing"] = torch.tensor(
            [self._str_2_idx[atom] for atom in gaff_types]
        )

        return g

    def _parametrize_gaff(self, g):
        from openmmforcefields.generators import SystemGenerator

        # define a system generator
        system_generator = SystemGenerator(
            small_molecule_forcefield=self.forcefield,
        )

        # create system
        sys = system_generator.create_system(
            topology=g.mol.to_topology().to_openmm(),
            molecules=g.mol,
        )


        for force in sys.getForces():
            name = force.__class__.__name__
            if "HarmonicBondForce" in name:
                assert force.getNumBonds() * 2 == g.heterograph.number_of_nodes(
                    "n2"
                )

                for idx in range(force.getNumBonds()):
                    idx0, idx1, eq, k = force.getBondParameters(idx)

                    print(idx0, idx1, eq, k)
                    # position = bond_lookup[(idx0, idx1)]
                    # _eq = (
                    #     g.nodes["n2"]
                    #     .data["eq%s" % suffix][position]
                    #     .detach()
                    #     .numpy()
                    #     .item()
                    # )
                    # _k = (
                    #     g.nodes["n2"]
                    #     .data["k%s" % suffix][position]
                    #     .detach()
                    #     .numpy()
                    #     .item()
                    # )
                    #
                    # _eq = Quantity(  # bond length
                    #     _eq, esp.units.DISTANCE_UNIT,
                    # ).value_in_unit(OPENMM_BOND_EQ_UNIT)
                    #
                    # _k = 2.0 * Quantity(  # bond force constant:
                    #     # since everything is enumerated twice in espaloma
                    #     # and once in OpenMM,
                    #     # we insert a coefficient of 2.0
                    #     _k,
                    #     esp.units.FORCE_CONSTANT_UNIT,
                    # ).value_in_unit(OPENMM_BOND_K_UNIT)
                    #
                    # force.setBondParameters(idx, idx0, idx1, _eq, _k)

            if "HarmonicAngleForce" in name:
                assert force.getNumAngles() * 2 == g.heterograph.number_of_nodes(
                    "n3"
                )

                for idx in range(force.getNumAngles()):
                    idx0, idx1, idx2, eq, k = force.getAngleParameters(idx)

                    print(idx0, idx1, idx2, eq, k)
                    # position = angle_lookup[(idx0, idx1, idx2)]
                    # _eq = (
                    #     g.nodes["n3"]
                    #     .data["eq%s" % suffix][position]
                    #     .detach()
                    #     .numpy()
                    #     .item()
                    # )
                    # _k = (
                    #     g.nodes["n3"]
                    #     .data["k%s" % suffix][position]
                    #     .detach()
                    #     .numpy()
                    #     .item()
                    # )
                    #
                    # _eq = Quantity(_eq, esp.units.ANGLE_UNIT,).value_in_unit(
                    #     OPENMM_ANGLE_EQ_UNIT
                    # )
                    #
                    # _k = 2.0 * Quantity(  # force constant
                    #     # since everything is enumerated twice in espaloma
                    #     # and once in OpenMM,
                    #     # we insert a coefficient of 2.0
                    #     _k,
                    #     esp.units.ANGLE_FORCE_CONSTANT_UNIT,
                    # ).value_in_unit(OPENMM_ANGLE_K_UNIT)
                    #
                    # force.setAngleParameters(idx, idx0, idx1, idx2, _eq, _k)

            if "PeriodicTorsionForce" in name:
                number_of_torsions = force.getNumTorsions()
                assert number_of_torsions <= g.heterograph.number_of_nodes("n4")

                # TODO: An alternative would be to start with an empty PeriodicTorsionForce and always call force.addTorsion
                #
                # if (
                #     "periodicity%s" % suffix not in g.nodes["n4"].data
                #     or "phase%s" % suffix not in g.nodes["n4"].data
                # ):
                #
                #     g.nodes["n4"].data["periodicity%s" % suffix] = torch.arange(
                #         1, 7
                #     )[None, :].repeat(g.heterograph.number_of_nodes("n4"), 1)
                #
                #     g.nodes["n4"].data["phases%s" % suffix] = torch.zeros(
                #         g.heterograph.number_of_nodes("n4"), 6
                #     )
                #
                # count_idx = 0
                # for idx in range(g.heterograph.number_of_nodes("n4")):
                #     idx0 = g.nodes["n4"].data["idxs"][idx, 0].item()
                #     idx1 = g.nodes["n4"].data["idxs"][idx, 1].item()
                #     idx2 = g.nodes["n4"].data["idxs"][idx, 2].item()
                #     idx3 = g.nodes["n4"].data["idxs"][idx, 3].item()
                #
                #     # assuming both (a,b,c,d) and (d,c,b,a) are listed for every torsion, only pick one of the orderings
                #     if idx0 < idx3:
                #         periodicities = g.nodes["n4"].data[
                #             "periodicity%s" % suffix
                #         ][idx]
                #         phases = g.nodes["n4"].data["phases%s" % suffix][idx]
                #         ks = g.nodes["n4"].data["k%s" % suffix][idx]
                #         for sub_idx in range(ks.flatten().shape[0]):
                #             k = ks[sub_idx].item()
                #             if k != 0.0:
                #                 _periodicity = periodicities[sub_idx].item()
                #                 _phase = phases[sub_idx].item()
                #
                #                 k = Quantity(
                #                     k, esp.units.ENERGY_UNIT,
                #                 ).value_in_unit(OPENMM_ENERGY_UNIT,)
                #
                #                 if count_idx < number_of_torsions:
                #                     force.setTorsionParameters(
                #                         # since everything is enumerated
                #                         # twice in espaloma
                #                         # and once in OpenMM,
                #                         # we insert a coefficient of 2.0
                #                         count_idx,
                #                         idx0,
                #                         idx1,
                #                         idx2,
                #                         idx3,
                #                         _periodicity,
                #                         _phase,
                #                         2.0 * k,
                #                     )
                #
                #                 else:
                #                     force.addTorsion(
                #                         # since everything is enumerated
                #                         # twice in espaloma
                #                         # and once in OpenMM,
                #                         # we insert a coefficient of 2.0
                #                         idx0,
                #                         idx1,
                #                         idx2,
                #                         idx3,
                #                         _periodicity,
                #                         _phase,
                #                         2.0 * k,
                #                     )
                #
                #                 count_idx += 1
                #
                #


    def _parametrize_smirnoff(self, g):
        # mol = self._convert_to_off(mol)

        forces = self.FF.label_molecules(g.mol.to_topology())[0]

        g.heterograph.apply_nodes(
            lambda node: {
                "k_ref": torch.Tensor(
                    [
                        forces["Bonds"][
                            tuple(node.data["idxs"][idx].numpy())
                        ].k.value_in_unit(esp.units.FORCE_CONSTANT_UNIT)
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                )[:, None]
            },
            ntype="n2",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "eq_ref": torch.Tensor(
                    [
                        forces["Bonds"][
                            tuple(node.data["idxs"][idx].numpy())
                        ].length.value_in_unit(esp.units.DISTANCE_UNIT)
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                )[:, None]
            },
            ntype="n2",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "k_ref": torch.Tensor(
                    [
                        forces["Angles"][
                            tuple(node.data["idxs"][idx].numpy())
                        ].k.value_in_unit(esp.units.ANGLE_FORCE_CONSTANT_UNIT)
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                )[:, None]
            },
            ntype="n3",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "eq_ref": torch.Tensor(
                    [
                        forces["Angles"][
                            tuple(node.data["idxs"][idx].numpy())
                        ].angle.value_in_unit(esp.units.ANGLE_UNIT)
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                )[:, None]
            },
            ntype="n3",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "epsilon_ref": torch.Tensor(
                    [
                        forces["vdW"][(idx,)].epsilon.value_in_unit(
                            esp.units.ENERGY_UNIT
                        )
                        for idx in range(g.heterograph.number_of_nodes("n1"))
                    ]
                )[:, None]
            },
            ntype="n1",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "sigma_ref": torch.Tensor(
                    [
                        forces["vdW"][(idx,)].rmin_half.value_in_unit(
                            esp.units.DISTANCE_UNIT
                        )
                        for idx in range(g.heterograph.number_of_nodes("n1"))
                    ]
                )[:, None]
            },
            ntype="n1",
        )

        def apply_torsion(node, n_max_phases=6):
            phases = torch.zeros(
                g.heterograph.number_of_nodes("n4"), n_max_phases,
            )

            periodicity = torch.zeros(
                g.heterograph.number_of_nodes("n4"), n_max_phases,
            )

            k = torch.zeros(g.heterograph.number_of_nodes("n4"), n_max_phases,)

            force = forces["ProperTorsions"]

            for idx in range(g.heterograph.number_of_nodes("n4")):
                idxs = tuple(node.data["idxs"][idx].numpy())
                if idxs in force:
                    _force = force[idxs]
                    for sub_idx in range(len(_force.periodicity)):
                        if hasattr(_force, "k%s" % sub_idx):
                            k[idx, sub_idx] = getattr(
                                _force, "k%s" % sub_idx
                            ).value_in_unit(esp.units.ENERGY_UNIT)

                            phases[idx, sub_idx] = getattr(
                                _force, "phase%s" % sub_idx
                            ).value_in_unit(esp.units.ANGLE_UNIT)

                            periodicity[idx, sub_idx] = getattr(
                                _force, "periodicity%s" % sub_idx
                            )

            return {
                "k_ref": k,
                "periodicity_ref": periodicity,
                "phases_ref": phases,
            }

        g.heterograph.apply_nodes(apply_torsion, ntype="n4")

        return g

    def _multi_typing_smirnoff(self, g):
        # mol = self._convert_to_off(mol)

        forces = self.FF.label_molecules(g.mol.to_topology())[0]

        g.heterograph.apply_nodes(
            lambda node: {
                "legacy_typing": torch.Tensor(
                    [
                        int(
                            forces["Bonds"][
                                tuple(node.data["idxs"][idx].numpy())
                            ].id[1:]
                        )
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                ).long()
            },
            ntype="n2",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "legacy_typing": torch.Tensor(
                    [
                        int(
                            forces["Angles"][
                                tuple(node.data["idxs"][idx].numpy())
                            ].id[1:]
                        )
                        for idx in range(node.data["idxs"].shape[0])
                    ]
                ).long()
            },
            ntype="n3",
        )

        g.heterograph.apply_nodes(
            lambda node: {
                "legacy_typing": torch.Tensor(
                    [
                        int(forces["vdW"][(idx,)].id[1:])
                        for idx in range(g.heterograph.number_of_nodes("n1"))
                    ]
                ).long()
            },
            ntype="n1",
        )

        return g

    def parametrize(self, g):
        """ Parametrize a molecular graph.

        """
        if "smirnoff" in self.forcefield or "openff" in self.forcefield:
            return self._parametrize_smirnoff(g)

        elif "gaff" in self.forcefield:
            return self._parametrize_gaff(g)

        else:
            raise NotImplementedError

    def typing(self, g):
        """ Type a molecular graph.

        """
        if "gaff" in self.forcefield:
            return self._type_gaff(g)

        else:
            raise NotImplementedError

    def multi_typing(self, g):
        """ Type a molecular graph for hetero nodes. """
        if "smirnoff" in self.forcefield:
            return self._multi_typing_smirnoff(g)

        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.typing(*args, **kwargs)
