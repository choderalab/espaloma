import espaloma as esp
import torch

def run():
    # grab dataset
    esol = esp.data.esol(first=20)

    # do some typing
    param = esp.graphs.legacy_force_field.LegacyForceField('smirnoff99Frosst'
        ).parametrize
    esol.apply(param, in_place=True) # this modify the original data

    # split
    # NOTE:
    # I don't like torch-generic splitting function as it requires
    # specifically the volume of each partition and it is inconsistent
    # with the specification of __getitem__ method
    ds_tr, ds_te = esol.split([4, 1])

    # get a loader object that views this dataset in some way
    # using this specific flag the dataset turns into an iterator
    # that outputs loss function, per John's suggestion
    loader = ds_tr.view('graph', batch_size=2)

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn('GraphConv')

    # define a representation
    representation = esp.nn.Sequential(
            layer,
            [32, 'tanh', 32, 'tanh', 32, 'tanh']
    )

    # define a readout
    readout = esp.nn.readout.janossy.JanossyPooling(
        config=[32, 'tanh'],
        in_features=32)

    net = torch.nn.Sequential(
        representation,
        readout
    )

    exp = esp.TrainAndTest(
        ds_tr=loader,
        ds_te=loader,
        net=net,
        metrics_tr=[esp.metrics.GraphMetric(
            between=['k_ref', 'k'],
            level='n2',
            base_metric=torch.nn.MSELoss()
        )],
        metrics_te=[esp.metrics.GraphMetric(
            between=['k_ref', 'k'],
            level='n2',
            base_metric=esp.metrics.rmse
        )],
        n_epochs=100,
    )

    results = exp.run()

    print(esp.app.report.markdown(results))


if __name__ == '__main__':
    run()
