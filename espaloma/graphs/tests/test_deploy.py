import pytest
import espaloma as esp

def test_methane():
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    g = esp.Graph('CCCC')
    g = ff.parametrize(g)
    print(g.mol)
    esp.graphs.deploy.openmm_system_from_graph(g, suffix="_ref")
