import pytest


def test_save_and_load():
    import espaloma as esp

    g = esp.Graph("C")
    ds = esp.data.dataset.GraphDataset([g])

    # Temporary directory will be automatically cleaned up
    from espaloma.data.utils import make_temp_directory

    with make_temp_directory() as tmpdir:
        import os

        filename = os.path.join(tmpdir, "ds")

        ds.save(filename)
        new_ds = esp.data.dataset.GraphDataset.load(filename)
