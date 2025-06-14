import pytest


def test_import():
    import espaloma.data.qcarchive_utils


def test_get_graph():
    from espaloma.data import qcarchive_utils

    client = qcarchive_utils.get_client()
    collection, record_names = qcarchive_utils.get_collection(client)
    # The order records are received is not guaranteed, and can change if,
    # e.g., the underlying database ends up being replaced by a copy during a database migration.
    # as such we need to use a specific record name.
    records_names_for_testing = ['c1c2c(c(c(c1f)n3cc(c3)o)cl)n(cc(c2=o)c(=o)[o-])c4c(cc(c(n4)n)f)f-3', 'c1c2c(cc(c1f)n3ccncc3)n(cc(c2=o)c(=o)[o-])c4cc4-0']

    record_name = records_names_for_testing[0]
    assert record_name in record_names

    graph = qcarchive_utils.get_graph(collection, record_name)
    assert graph is not None


    graphs = qcarchive_utils.get_graphs(collection, records_names_for_testing)
    assert len(graphs) == 2
    assert graphs[0] is not None


def test_notsupported_dataset():
    from espaloma.data import qcarchive_utils

    name = "DBH24"
    collection_type = "reaction"
    collection, record_names = qcarchive_utils.get_collection(
        qcarchive_utils.get_client("ml.qcarchive.molssi.org"), collection_type, name
    )
    record_name = record_names[0]

    with pytest.raises(Exception):
        graph = qcarchive_utils.get_graph(collection, record_name, spec_name="spec_2")


def test_get_torsiondrive():
    from espaloma.data import qcarchive_utils
    import numpy as np

    record_name = "[h]c1c(c(c(c([c:1]1[n:2]([c:3](=[o:4])c(=c([h])[h])[h])c([h])([h])[h])[h])[h])n(=o)=o)[h]"

    # example dataset 
    name = "OpenFF Amide Torsion Set v1.0"
    collection_type = "torsiondrive"

    collection, record_names = qcarchive_utils.get_collection(
        qcarchive_utils.get_client(), collection_type, name
    )
    record_info = collection.get_record(record_name, specification_name="default")

    (
        flat_angles,
        xyz_in_order,
        energies_in_order,
        gradients_in_order,
    ) = qcarchive_utils.fetch_td_record(record_info)

    assert flat_angles.shape == (24,)
    assert energies_in_order.shape == (24,)
    assert gradients_in_order.shape == (24, 25, 3)
    assert xyz_in_order.shape == (24, 25, 3)

    assert np.isclose(energies_in_order[0], -722.2850260791969)
    assert np.all(
        flat_angles
        == np.array(
            [
                -165,
                -150,
                -135,
                -120,
                -105,
                -90,
                -75,
                -60,
                -45,
                -30,
                -15,
                0,
                15,
                30,
                45,
                60,
                75,
                90,
                105,
                120,
                135,
                150,
                165,
                180,
            ]
        )
    )
    assert np.allclose(
        xyz_in_order[0][0], np.array([-0.66407807, -8.59922225, -0.02685972])
    )
