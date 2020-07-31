import pytest


@pytest.fixture
def baseline():
    import espaloma as esp
    g = esp.Graph('c1ccccc1')


    # get force field
    forcefield = esp.graphs.legacy_force_field.LegacyForceField(
        'smirnoff99Frosst'
    )

    # param / typing
    operation = forcefield.parametrize

    operation(g)

    baseline = esp.nn.baselines.FreeParameterBaseline(g_ref=g.heterograph)

    return baseline

def test_init(baseline):
    baseline


def test_parameter(baseline):
    print(list(baseline.parameters()))

    assert len(list(baseline.parameters())) > 0
