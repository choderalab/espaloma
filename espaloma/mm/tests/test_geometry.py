import pytest
import torch
import espaloma as esp

def test_import():
    esp.mm.geometry

@pytest.fixture
def g():
    g = esp.Graph('c1ccccc1')
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=10, n_steps_per_sample=10
    )
    g = simulation.run(g, in_place=True)
    return g

def test_geometry_all(g):
    g = esp.mm.geometry.geometry_in_graph(g.heterograph)
    assert g.nodes['n2'].data['x'].shape == torch.Size([24, 10])
