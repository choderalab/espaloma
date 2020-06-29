import pytest
import espaloma as esp


def test_smirnoff():
    ff = esp.graphs.legacy_force_field.LegacyForceField("smirnoff99Frosst")
    mol = esp.data.esol(first=1)[0].mol
    mol = ff.parametrize(mol)
