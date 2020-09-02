import torch

import espaloma as esp


def run(layer_name):
    # grab dataset
    # esol = esp.data.esol(first=20)
    ds = esp.data.zinc(first=1000).shuffle()

    # do some typing
    typing = esp.graphs.legacy_force_field.LegacyForceField('gaff-1.81')
    ds.apply(typing, in_place=True) # this modify the original data

    # split
    # NOTE:
    # I don't like torch-generic splitting function as it requires
    # specifically the volume of each partition and it is inconsistent
    # with the specification of __getitem__ method
    ds_tr, ds_te = ds.split([4, 1])

    ds_tr.save('ds_tr.th')
    ds_te.save('ds_te.th')

    # get a loader object that views this dataset in some way
    # using this specific flag the dataset turns into an iterator
    # that outputs loss function, per John's suggestion
    ds_tr = ds_tr.view('graph', batch_size=20)
    ds_te = ds_te.view('graph', batch_size=len(ds_te))

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn(layer_name)

    # define a representation
    representation = esp.nn.Sequential(
            layer,
            [32, 'leaky_relu', 32, 'leaky_relu', 32, 'leaky_relu']
    )

    # define a readout
    readout = esp.nn.readout.node_typing.NodeTyping(
            in_features=32,
            n_classes=100
    ) # not too many elements here I think?

    net = torch.nn.Sequential(
        representation,
        readout
    )

    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_te,
        net=net,
        metrics_te=[esp.metrics.TypingAccuracy()],
        n_epochs=500,
    )

    results = exp.run()

    print(esp.app.report.markdown(results))

    import pickle
    with open(layer_name + "_ref_g_test.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_test, f_handle)

    with open(layer_name + "_ref_g_training.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_training, f_handle)


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
