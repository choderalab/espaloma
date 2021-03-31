import pytest

def test_linear_combination():
    import torch
    import espaloma as esp
    assert esp.mm.functional.linear_mixture(
        0.0,
        torch.tensor([[0.0, 0.0]]),
    ) == 0.0
    assert esp.mm.functional.linear_mixture(
        1.0,
        torch.tensor([[1.0, 1.0]]),
        [0.0, 2.0],
    ) == 1.0

def test_consistency():
    import torch
    import espaloma as esp
    g = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(n_samples=10, n_steps_per_sample=10)
    g = simulation.run(g, in_place=True)

    g.nodes['n2'].data['coefficients'] = torch.randn(
        g.heterograph.number_of_nodes("n2"), 2
    ).exp()

    g.nodes['n3'].data['coefficients'] = torch.randn(
        g.heterograph.number_of_nodes("n3"), 2
    ).exp()

    esp.mm.geometry.geometry_in_graph(g.heterograph)

    esp.mm.energy.energy_in_graph(g.heterograph, terms=['n2', 'n3'])

    u0_2 = g.nodes['n2'].data['u'] - g.nodes['n2'].data['u'].mean(dim=1, keepdims=True)
    u0_3 = g.nodes['n3'].data['u'] - g.nodes['n3'].data['u'].mean(dim=1, keepdims=True)
    u0 = g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=1, keepdims=True)

    g.nodes['n2'].data['k'], g.nodes['n2'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
        g.nodes['n2'].data['coefficients'][:, 0][:, None],
        g.nodes['n2'].data['coefficients'][:, 1][:, None],
        1.5, 6.0,
    )

    import math
    g.nodes['n3'].data['k'], g.nodes['n3'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
        g.nodes['n3'].data['coefficients'][:, 0][:, None],
        g.nodes['n3'].data['coefficients'][:, 1][:, None],
        0.0, math.pi,
    )

    g.nodes['n2'].data.pop('coefficients')
    g.nodes['n3'].data.pop('coefficients')

    esp.mm.energy.energy_in_graph(g.heterograph, terms=['n2', 'n3'])

    u1_2 = g.nodes['n2'].data['u'] - g.nodes['n2'].data['u'].mean(dim=1, keepdims=True)
    u1_3 = g.nodes['n3'].data['u'] - g.nodes['n3'].data['u'].mean(dim=1, keepdims=True)
    u1 = g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=1, keepdims=True)

    import numpy.testing as npt

    npt.assert_almost_equal(
        u0_2.detach().numpy(), u1_2.detach().numpy(),
        decimal=3,
    )

    npt.assert_almost_equal(
        u0_3.detach().numpy(), u1_3.detach().numpy(),
        decimal=3,
    )

    npt.assert_almost_equal(
        u0.detach().numpy(), u1.detach().numpy(),
        decimal=3,
    )
