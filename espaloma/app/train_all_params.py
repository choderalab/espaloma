# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import numpy as np
import torch

import espaloma as esp


def run(args):
    # define data
    data = getattr(esp.data, args.data)(first=args.first)

    # get force field
    forcefield = esp.graphs.legacy_force_field.LegacyForceField(
        args.forcefield
    )

    # param / typing
    operation = forcefield.parametrize

    # apply to dataset
    data = data.apply(operation, in_place=True)

    # split
    partition = [int(x) for x in args.partition.split(":")]
    ds_tr, ds_te = data.split(partition)

    # batch
    ds_tr = ds_tr.view("graph", batch_size=args.batch_size)
    ds_te = ds_te.view("graph", batch_size=args.batch_size)

    # layer
    layer = esp.nn.layers.dgl_legacy.gn(args.layer)

    # representation
    representation = esp.nn.Sequential(layer, config=args.config)

    # get the last bit of units
    units = [x for x in args.config if isinstance(x, int)][-1]

    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units,
        config=args.janossy_config,
        out_features={
            2: ["k", "eq"],
            3: ["k", "eq"],
        },
    )

    net = torch.nn.Sequential(representation, readout)

    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=torch.nn.L1Loss(),
            between=[param, param + "_ref"],
            level=term,
        )
        for param in ["k", "eq"]
        for term in ["n2", "n3"]
    ]

    metrics_te = [
        esp.metrics.GraphMetric(
            base_metric=base_metric,
            between=[param, param + "_ref"],
            level=term,
        )
        for param in ["k", "eq"]
        for term in ["n2", "n3"]
        for base_metric in [esp.metrics.rmse, esp.metrics.r2]
    ]

    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_te,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
    )

    results = exp.run()

    print(esp.app.report.markdown(results))

    import os

    os.mkdir(args.out)

    with open(args.out + "/architecture.txt", "w") as f_handle:
        f_handle.write(str(exp))

    with open(args.out + "/result_table.md", "w") as f_handle:
        f_handle.write(esp.app.report.markdown(results))

    curves = esp.app.report.curve(results)

    for spec, curve in curves.items():
        np.save(args.out + "/" + "_".join(spec) + ".npy", curve)

    import pickle

    with open(args.out + "/ref_g_test.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_test, f_handle)

    with open(args.out + "/ref_g_training.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_training, f_handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="alkethoh", type=str)
    parser.add_argument("--out", default="results", type=str)
    parser.add_argument("--first", default=-1, type=int)
    parser.add_argument("--partition", default="4:1", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument(
        "--forcefield", default="smirnoff99Frosst-1.1.0", type=str
    )
    parser.add_argument("--layer", default="GraphConv", type=str)
    parser.add_argument("--n_classes", default=100, type=int)
    parser.add_argument(
        "--config", nargs="*", default=[32, "tanh", 32, "tanh", 32, "tanh"]
    )

    parser.add_argument(
        "--training_metrics", nargs="*", default=["TypingCrossEntropy"]
    )
    parser.add_argument(
        "--test_metrics", nargs="*", default=["TypingAccuracy"]
    )

    parser.add_argument("--janossy_config", nargs="*", default=[32, "tanh"])

    parser.add_argument("--n_epochs", default=10, type=int)

    args = parser.parse_args()

    run(args)
