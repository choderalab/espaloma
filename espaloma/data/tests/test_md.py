import pytest
import torch


def test_init():
    import espaloma.data.md

@pytest.fixture
def graph():
    import espaloma as esp
    graph = esp.Graph('c1ccccc1')
    return graph

@pytest.fixture
def ds():
    import espaloma as esp
    ds = esp.data.esol(first=10)
    return ds


def test_system(graph):
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation()

def test_run(graph):
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=10, n_steps_per_sample=10
    )

    samples = simulation.run(graph, in_place=False)

    assert samples.shape == torch.Size([10, 12, 3])

def test_run_in_place(graph):
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=10, n_steps_per_sample=10
    )

    graph = simulation.run(graph, in_place=True)

    assert graph.nodes['n1'].data['xyz'].shape == torch.Size([12, 10, 3])

def test_apply(ds):
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=1, n_steps_per_sample=1
    ).run

    ds.apply(simulation, in_place=True)

    assert ds.graphs[0].nodes['n1'].data['xyz'].shape[-1] == 3
    assert ds.graphs[0].nodes['n1'].data['xyz'].shape[-2] == 1
