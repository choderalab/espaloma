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
    data = getattr(esp.data, args.data)(first=1)

    # get force field
    forcefield = esp.graphs.legacy_force_field.LegacyForceField(
        args.forcefield
    )

    # param / typing
    operation = forcefield.parametrize

    # apply to dataset
    data = data.apply(operation, in_place=True)

    # apply simulation
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=1000, n_steps_per_sample=10
    )

    data = data.apply(simulation.run, in_place=True)

    # only one bit of data
    ds = data.view("graph", batch_size=1)
    ds_te = ds_tr = ds

    for g in ds:
        pass

    # representation
    representation = esp.nn.baselines.FreeParameterBaseline(g_ref=g)

    net = torch.nn.Sequential(
            representation, 
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(),
            esp.mm.energy.EnergyInGraph(suffix='_ref'),
    )

    optimizer = torch.optim.LBFGS(
        net.parameters(),
        0.01,
        line_search_fn='strong_wolfe'
    )


    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(),
            between=['u', 'u_ref'],
            level='g'
        )
    ]

    metrics_te = [
        esp.metrics.GraphMetric(
            base_metric=base_metric,
            between=[param, param + '_ref'],
            level=term
        ) for param in ['u'] for term in ['g']
        for base_metric in [
            esp.metrics.rmse,
            esp.metrics.r2
        ]
    ]


    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_te,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        normalize=esp.data.normalize.PositiveNotNormalize,
        optimizer=optimizer,
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


    print(esp.app.report.markdown(results))

    import pickle
    with open(args.out + "/ref_g_test.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_test, f_handle)

    with open(args.out + "/ref_g_training.th", "wb") as f_handle:
        pickle.dump(exp.ref_g_training, f_handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="alkethoh", type=str)
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
    parser.add_argument(
        "--out", default="results", type=str
    )
    parser.add_argument("--janossy_config", nargs="*", default=[32, "tanh"])

    parser.add_argument("--n_epochs", default=10, type=int)

    args = parser.parse_args()

    run(args)
