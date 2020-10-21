import pytest


def test_import():
    import espaloma.data.qcarchive_utils


def test_get_graph():
    from espaloma.data import qcarchive_utils

    client = qcarchive_utils.get_client()
    collection, record_names = qcarchive_utils.get_collection(client)
    record_name = record_names[0]
    graph = qcarchive_utils.get_graph(collection, record_name)
