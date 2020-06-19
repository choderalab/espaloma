import pytest
import torch

def test_import():
    import espaloma as esp
    esp.app.experiment


@pytest.fixture
def data():
    import espaloma as esp

    esol = esp.data.esol(first=20)

    # do some typing
    typing = esp.graphs.legacy_force_field.LegacyForceField('gaff-1.81')
    esol.apply(typing, in_place=True) # this modify the original data

    return esol

@pytest.fixture
def net():
    import espaloma as esp

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn('GraphConv')

    # define a representation
    representation = esp.nn.Sequential(
            layer,
            [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    # define a readout
    readout = esp.nn.readout.node_typing.NodeTyping(
            in_features=32,
            n_classes=100) # not too many elements here I think?

    net = torch.nn.Sequential(
        representation,
        readout)

    return net


def test_data_and_net(data, net):
    data
    net


@pytest.fixture
def train(data, net):
    import espaloma as esp
    train = esp.app.experiment.Train(
        net=net,
        data=data,
        n_epochs=1,
        metrics=[esp.metrics.GraphMetric(
            base_metric=torch.nn.CrossEntropyLoss(),
            between=['nn_typing', 'legacy_typing'])])

    return train

def test_train(train):
    train.train()

def test_test(train, net, data):
    import espaloma as esp
    train.train()
    test = esp.app.experiment.Test(
        net=net,
        data=data,
        states=train.states)


def test_train_and_test(net, data):
    import espaloma as esp

    train_and_test = esp.app.experiment.TrainAndTest(
        net=net,
        n_epochs=1,
        ds_tr=data,
        ds_te=data
    )
