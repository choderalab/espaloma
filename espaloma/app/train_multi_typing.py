# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import espaloma as esp
import os
import numpy as np
import torch

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
            1: ['nn_typing'],
            2: ['nn_typing'],
            3: ['nn_typing']
        }
    )

    net = torch.nn.Sequential(representation, readout)

    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=torch.nn.CrossEntropyLoss(),
            between=['nn_typing', 'legacy_typing'],
            level=term,
        ) for term in ['n2', 'n3']
    ]

    metrics_te = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.accuracy,
            between=['nn_typing', 'legacy_typing'],
            level=term
        )  for term in ['n2', 'n3']
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="alkethoh", type=str)
    parser.add_argument("--first", default=-1, type=int)
    parser.add_argument("--partition", default="4:1", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--forcefield", default="smirnoff99Frosst", type=str)
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
