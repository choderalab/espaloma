import numpy.testing as npt
import pytest


def test_import():
    from espaloma.data.normalize import BaseNormalize


def test_normalize_esol():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField(
                "smirnoff99Frosst-1.1.0"
            ).parametrize,
            in_place=True,
        )
    )


def test_log_normalize_esol():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetLogNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField(
                "smirnoff99Frosst-1.1.0"
            ).parametrize,
            in_place=True,
        )
    )


def test_normal_normalize_reproduce():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField(
                "smirnoff99Frosst-1.1.0"
            ).parametrize,
            in_place=True,
        )
    )

    esol = esp.data.esol(first=1)

    # do some typing
    param = esp.graphs.legacy_force_field.LegacyForceField(
        "smirnoff99Frosst-1.1.0"
    ).parametrize
    esol.apply(param, in_place=True)  # this modify the original data

    g = esol[0]

    import copy

    g_ = copy.deepcopy(g)

    g = normalize.norm(g)

    g.nodes["n2"].data["k"] = g.nodes["n2"].data["k_ref"]
    g.nodes["n2"].data["eq"] = g.nodes["n2"].data["eq_ref"]

    g = normalize.unnorm(g)

    npt.assert_almost_equal(
        g.nodes["n2"].data["k"].detach().numpy(),
        g_.nodes["n2"].data["k_ref"].detach().numpy(),
    )

    npt.assert_almost_equal(
        g.nodes["n2"].data["eq"].detach().numpy(),
        g_.nodes["n2"].data["eq_ref"].detach().numpy(),
    )


def test_log_normal_normalize_reproduce():
    import espaloma as esp

    normalize = esp.data.normalize.DatasetLogNormalNormalize(
        dataset=esp.data.esol(first=10).apply(
            esp.graphs.legacy_force_field.LegacyForceField(
                "smirnoff99Frosst-1.1.0"
            ).parametrize,
            in_place=True,
        )
    )

    esol = esp.data.esol(first=1)

    # do some typing
    param = esp.graphs.legacy_force_field.LegacyForceField(
        "smirnoff99Frosst-1.1.0"
    ).parametrize
    esol.apply(param, in_place=True)  # this modify the original data

    g = esol[0]

    import copy

    g_ = copy.deepcopy(g)

    g = normalize.norm(g)

    g.nodes["n2"].data["k"] = g.nodes["n2"].data["k_ref"]
    g.nodes["n2"].data["eq"] = g.nodes["n2"].data["eq_ref"]

    g = normalize.unnorm(g)

    npt.assert_almost_equal(
        g.nodes["n2"].data["k"].detach().numpy(),
        g_.nodes["n2"].data["k_ref"].detach().numpy(),
        decimal=1,
    )

    npt.assert_almost_equal(
        g.nodes["n2"].data["eq"].detach().numpy(),
        g_.nodes["n2"].data["eq_ref"].detach().numpy(),
        decimal=1,
    )
