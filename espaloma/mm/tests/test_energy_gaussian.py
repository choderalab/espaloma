import pytest

def test_energy():
    import espaloma as esp
    import torch

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
            in_features=32, config=[32, "tanh"],
            out_features={
                1: {'sigma': 1, 'epsilon': 1},
                2: {'coefficients': 200},
                3: {'k':1, 'eq': 1},
            },
        ),
    )

    g = net(g.heterograph)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g)
    esp.mm.energy.energy_in_graph(g)

    esp.mm.energy.energy_in_graph(g, suffix="_ref")
