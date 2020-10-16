import pytest
import torch

import espaloma as esp


def test_import():
    esp.mm.geometry


# later, if we want to do multiple molecules, group these into a struct
smiles = "c1ccccc1"
n_samples = 2
expected_n_terms = dict(n2=24, n3=36, n4=48, n4_improper=36)


@pytest.fixture
def g():
    g = esp.Graph(smiles)
    from espaloma.data.md import MoleculeVacuumSimulation

    simulation = MoleculeVacuumSimulation(n_samples=n_samples, n_steps_per_sample=1)
    g = simulation.run(g, in_place=True)
    return g


def test_geometry_can_be_computed_without_exceptions(g):
    g = esp.mm.geometry.geometry_in_graph(g.heterograph)


def test_geometry_n_terms(g):
    g = esp.mm.geometry.geometry_in_graph(g.heterograph)

    for term in ["n2", "n3", "n4", "n4_improper"]:
        assert g.nodes[term].data["x"].shape == torch.Size([expected_n_terms[term], n_samples])
