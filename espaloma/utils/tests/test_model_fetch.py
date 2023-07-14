import espaloma as esp
import torch
from espaloma.utils.model_fetch import get_model, get_model_path
from openff.toolkit.topology import Molecule


def test_get_model_path(tmp_path):
    model_dir = tmp_path / "latest"
    model_path = get_model_path(model_dir=model_dir, disable_progress_bar=True)

    molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    molecule_graph = esp.Graph(molecule)

    espaloma_model = torch.load(model_path)
    espaloma_model.eval()
    espaloma_model(molecule_graph.heterograph)


def test_get_model(tmp_path):
    model_dir = tmp_path / "zoo"

    espaloma_model = get_model()

    molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    molecule_graph = esp.Graph(molecule)
    espaloma_model(molecule_graph.heterograph)
