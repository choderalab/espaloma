import pytest

import espaloma as esp


def test_smirnoff_esol_first():
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst-1.1.0")
    g = esp.data.esol(first=1)[0]
    g = ff.parametrize(g)

# def test_smirnoff_strange_mol():
#     ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst-1.1.0")
#     g = esp.Graph(
#         "[H]c1c(nc(n(=O)c1N([H])[H])N([H])[H])N2C(C(C(C(C2([H])[H])([H])[H])([H])[H])([H])[H])([H])[H]"
#     )
#     g = ff.parametrize(g)
#
#
# def test_multi_typing():
#     ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst-1.1.0")
#     g = esp.data.esol(first=1)[0]
#     g = ff.multi_typing(g)
