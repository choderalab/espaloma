import pytest
import torch
import espaloma as esp

def test_import():
    esp.mm.energy

def test_energy():
    g = esp.Graph('c1ccccc1')

    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=10, n_steps_per_sample=10
    )
    g = simulation.run(g, in_place=True)

    # parametrize
    layer = esp.nn.dgl_legacy.gn()
    net = torch.nn.Sequential(
        esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"]),
        esp.nn.readout.janossy.JanossyPooling(
            in_features=32,
            config=[32, 'tanh']),
    )

    g = net(g.heterograph)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g)
    esp.mm.energy.energy_in_graph(g)
