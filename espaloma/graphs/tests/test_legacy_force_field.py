import pytest


def test_import():
    import espaloma as esp

    gaff = esp.graphs.LegacyForceField("gaff-1.81")


def test_translation_dict():
    import espaloma as esp

    gaff = esp.graphs.LegacyForceField("gaff-1.81")

    assert isinstance(gaff._str_2_idx, dict)
    assert isinstance(gaff._idx_2_str, dict)


def test_type_benzene():

    import espaloma as esp

    from rdkit import Chem

    m = Chem.MolFromSmiles("c1ccccc1")

    gaff = esp.graphs.LegacyForceField("gaff-1.81")

    g = gaff.typing(m)

    print(g)
