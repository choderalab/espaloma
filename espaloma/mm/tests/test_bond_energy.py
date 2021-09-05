import pytest

def test_multiple_conformation():
    import espaloma as esp

    g = esp.Graph('c1ccccc1')

    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=10, n_steps_per_sample=10
    )
    g = simulation.run(g, in_place=True)

    param = esp.graphs.legacy_force_field.LegacyForceField(
        'smirnoff99Frosst-1.1.0').parametrize

    g = param(g)

    esp.mm.geometry.geometry_in_graph(g.heterograph)

    esp.mm.energy.energy_in_graph(g.heterograph, suffix='_ref')
