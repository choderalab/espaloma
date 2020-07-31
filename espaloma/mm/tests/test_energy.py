import pytest
import torch

import espaloma as esp


def test_import():
    esp.mm.energy


def test_energy():
    g = esp.Graph("c1ccccc1")

    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    simulation = MoleculeVacuumSimulation(n_samples=10, n_steps_per_sample=10)
    g = simulation.run(g, in_place=True)

    param = esp.graphs.legacy_force_field.LegacyForceField(
        "smirnoff99Frosst"
    ).parametrize

    g = param(g)

    # parametrize
    layer = esp.nn.dgl_legacy.gn()
    net = torch.nn.Sequential(
        esp.nn.Sequential(layer, [32, "tanh", 32, "tanh", 32, "tanh"]),
        esp.nn.readout.janossy.JanossyPooling(
            in_features=32, config=[32, "tanh"]
        ),
    )

    g = net(g.heterograph)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g)
    esp.mm.energy.energy_in_graph(g)

    esp.mm.energy.energy_in_graph(g, suffix="_ref")


def test_energy_consistent():
    g = esp.Graph("c1ccccc1")

    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    simulation = MoleculeVacuumSimulation(n_samples=10, n_steps_per_sample=10)
    g = simulation.run(g, in_place=True)

    param = esp.graphs.legacy_force_field.LegacyForceField(
        "smirnoff99Frosst"
    ).parametrize

    g = param(g)

    for node in ["n1", "n2", "n3"]:
        _dict = {}
        for data in g.nodes[node].data.keys():
            if data.endswith("_ref"):
                _dict[data.replace("_ref", "")] = g.nodes[node].data[data]
        for key, value in _dict.items():
            g.nodes[node].data[key] = value

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph)

    esp.mm.energy.energy_in_graph(g.heterograph, suffix="_ref")
