import io
import json
import pytest
import shutil
import importlib_resources
import espaloma as esp


def test_graph():
    import espaloma as esp

    g = esp.Graph("c1ccccc1")

    print(g.heterograph)


@pytest.fixture
def graph():
    import espaloma as esp

    return esp.Graph("c1ccccc1")


def test_ndata_consistency(graph):
    import torch

    assert torch.equal(graph.ndata["h0"], graph.nodes["n1"].data["h0"])


@pytest.mark.parametrize(
    "molecule, charge",
    [
        pytest.param("C", 0, id="methane"),
        pytest.param("[NH4+]", 1, id="Ammonium"),
        pytest.param("CC(=O)[O-]", -1, id="Acetate"),
    ],
)
def test_formal_charge(molecule, charge):
    import espaloma as esp

    graph = esp.Graph(molecule)
    assert graph.nodes["g"].data["sum_q"].numpy()[0] == charge


def test_save_and_load(graph):
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        graph.save(tempdir + "/g.esp")
        new_graph = esp.Graph()
        new_graph.load(tempdir + "/g.esp")

    assert graph.homograph.number_of_nodes == graph.homograph.number_of_nodes

    assert graph.homograph.number_of_edges == graph.homograph.number_of_edges

def test_load_from_older_openff(tmp_path_factory):
    """Tests creating a graph from a json-serialized mol with older openff-toolkit
    version (0.10.x)

    This checks that the serialized molecule doesn't have the expected hierarchy_schemes
    key, which will be created on the fly when loaded as a graph.

    This tests creates a graph with
    """
    # Load json serialized off 0.10.6 molecule and save it in path
    from openff.toolkit import Molecule
    mol_json_path = importlib_resources.files('espaloma.data') / 'off-mol_0_10_6.json'
    with open(str(mol_json_path), "r") as json_file:
        # This loads it as a string -- seems like an off toolkit limitation
        mol_json_str = json.load(json_file)
    mol_dict = json.load(io.StringIO(mol_json_str))
    assert "hierarchy_schemes" not in mol_dict, "Serialized json mol contains unexpected key."
    # Save json molecule in path
    out_esp_dir_1 = tmp_path_factory.mktemp("esp1")
    shutil.copy(mol_json_path, out_esp_dir_1 / "mol.json")

    # update dicitonary and create espaloma graph with the same molecule
    mol_dict["hierarchy_schemes"] = dict()
    off_molecule = Molecule.from_dict(mol_dict)
    smiles = off_molecule.to_smiles()
    g = esp.Graph(smiles)
    # Save the graph
    out_esp_dir_2 = tmp_path_factory.mktemp("esp2") / "esp-test"
    g.save(str(out_esp_dir_2))
    # copy homo/hetero-graphs to original dir
    shutil.copy(out_esp_dir_2 / "homograph.bin", out_esp_dir_1)
    shutil.copy(out_esp_dir_2 / "heterograph.bin", out_esp_dir_1)

    # Load espaloma from original directory -- with mol serialized from off 0.10.6
    esp_graph = esp.Graph.load(str(out_esp_dir_1))

    assert esp_graph.mol == g.mol, f"Read molecule from esp graph, {esp_graph.mol} is not " \
                                   f"the same as the expected molecule {off_molecule}."


# TODO: test offmol_indices
# TODO: test relationship_indices_from_offmol
