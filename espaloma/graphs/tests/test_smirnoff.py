import pytest
import espaloma as esp


def test_smirnoff_esol_first():
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    mol = esp.data.esol(first=1)[0].mol
    mol = ff.parametrize(mol)


def test_smirnoff_strange_mol():
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    mol = esp.Graph('[H]c1c(nc(n(=O)c1N([H])[H])N([H])[H])N2C(C(C(C(C2([H])[H])([H])[H])([H])[H])([H])[H])([H])[H]').mol
    mol = ff.parametrize(mol)
