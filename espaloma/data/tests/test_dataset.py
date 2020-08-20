import pytest


def test_tiny_dataset():
    import espaloma as esp

    xs = list(range(5))
    ds = esp.data.dataset.Dataset(xs)


@pytest.fixture
def ds():
    xs = list(range(5))
    import espaloma as esp

    return esp.data.dataset.Dataset(xs)


def test_get(ds):
    assert ds[0] == 0


def test_len(ds):
    assert len(ds) == 5


def test_iter(ds):
    assert all(x == x_ for (x, x_) in zip(ds, range(5)))


def test_slice(ds):
    import espaloma as esp

    sub_ds = ds[:2]
    assert isinstance(ds, esp.data.dataset.Dataset)
    assert len(sub_ds) == 2


def test_split(ds):
    a, b = ds.split([1, 4])
    assert len(a) == 1
    assert len(b) == 4


@pytest.fixture
def ds_new(ds):
    fn = lambda x: x + 1
    return ds.apply(fn)


def test_no_change(ds_new):
    assert all(x == x_ for (x, x_) in zip(ds_new.graphs, range(5)))


def test_get_new(ds_new):
    assert ds_new[0] == 1


def test_len_new(ds_new):
    assert len(ds_new) == 5


def test_iter_new(ds_new):
    assert all(x == x_ + 1 for (x, x_) in zip(ds_new, range(5)))


@pytest.fixture
def ds_newer(ds):
    fn = lambda x: x + 1
    return ds.apply(fn).apply(fn)


def test_iter_newer(ds_newer):
    assert all(x == x_ + 2 for (x, x_) in zip(ds_newer, range(5)))


def test_no_return(ds):
    fn = lambda x: x + 1
    ds.apply(fn).apply(fn)
    assert all(x == x_ + 2 for (x, x_) in zip(ds, range(5)))
