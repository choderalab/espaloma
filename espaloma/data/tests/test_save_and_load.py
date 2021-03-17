import pytest


def test_save_and_load():
    import espaloma as esp
    g = esp.Graph('C')
    ds = esp.data.dataset.GraphDataset([g])
    ds.save('ds')

    new_ds = esp.data.dataset.GraphDataset.load('ds')

    os.rmdir('ds')
