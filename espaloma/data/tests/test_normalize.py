import pytest

def test_import():
    from espaloma.data.normalize import BaseNormalize

def test_normalize_esol():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField('smirnoff99Frosst'
                    ).parametrize,
            in_place=True
        )
    )

def test_log_normalize_esol():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetLogNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField('smirnoff99Frosst'
                    ).parametrize,
            in_place=True
        )
    )
