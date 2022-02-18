import pytest
import torch

import espaloma as esp
from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers

def test_import():
    esp.mm.geometry


# later, if we want to do multiple molecules, group these into a struct
smiles = "c1ccccc1"
n_samples = 2
## Different number of expected terms for different improper permutations
expected_n_terms = {
    'none': dict(n2=24, n3=36, n4=48, n4_improper=36),
    'espaloma': dict(n2=24, n3=36, n4=48, n4_improper=36),
    'smirnoff': dict(n2=24, n3=36, n4=48, n4_improper=18)
}

@pytest.fixture
def all_g():
    from espaloma.data.md import MoleculeVacuumSimulation

    all_g = {}
    for improper_def in expected_n_terms.keys():
        g = esp.Graph(smiles)
        if improper_def != 'none':
            regenerate_impropers(g, improper_def)

        simulation = MoleculeVacuumSimulation(
            n_samples=n_samples, n_steps_per_sample=1
        )
        g = simulation.run(g, in_place=True)
        all_g[improper_def] = g
    return all_g


def test_geometry_can_be_computed_without_exceptions(all_g):
    for g in all_g.values():
        g = esp.mm.geometry.geometry_in_graph(g.heterograph)


def test_geometry_n_terms(all_g):
    for improper_def, g in all_g.items():
        g = esp.mm.geometry.geometry_in_graph(g.heterograph)

        for term, n_terms in expected_n_terms[improper_def].items():
            assert g.nodes[term].data["x"].shape == torch.Size(
                [n_terms, n_samples]
            )

