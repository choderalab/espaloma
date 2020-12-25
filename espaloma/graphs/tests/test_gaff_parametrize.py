import pytest
import espaloma as esp

def test_gaff_parametrize():
    ff = esp.graphs.legacy_force_field.LegacyForceField("gaff-1.81")
    g = esp.Graph(
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    )
    ff.parametrize(g)

    print(g.nodes['n2'].data)
    print(g.nodes['n3'].data)
