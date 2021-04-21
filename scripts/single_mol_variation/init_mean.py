# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os

import numpy as np
import torch

import espaloma as esp

def run(args):
    # define data
    g = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    data = esp.data.dataset.GraphDataset([g])

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
        n_samples=500, n_steps_per_sample=100,
    )

    data = data.apply(simulation.run, in_place=True)

    # batch
    ds = data.view("graph", batch_size=1)

    g = next(iter(ds))


    
    class AddMean(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['k'] = g.nodes['n2'].data['k_ref'].detach().mean() + g.nodes['n2'].data['k']
            g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq_ref'].detach().mean() + g.nodes['n2'].data['eq']
            g.nodes['n3'].data['k'] = g.nodes['n3'].data['k_ref'].detach().mean() + g.nodes['n3'].data['k']
            g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq_ref'].detach().mean() + g.nodes['n3'].data['eq']
            
            return g

    add_mean = AddMean()

    if args.layer != "Free":
        # layer
        layer = esp.nn.layers.dgl_legacy.gn(args.layer)

        # representation
        representation = esp.nn.Sequential(layer, config=args.config)

        # get the last bit of units
        units = [int(x) for x in args.config if isinstance(x, int) or isinstance(x, str) and x.isdigit()][-1]

        janossy_config = []
        for x in args.janossy_config:
            if isinstance(x, int):
                janossy_config.append(int(x))

            elif x.isdigit():
                janossy_config.append(int(x))

            else:
                janossy_config.append(x)

        readout = esp.nn.readout.janossy.JanossyPooling(
            in_features=units, config=janossy_config,
        )

        net = torch.nn.Sequential(
                representation, 
                readout,
                esp.mm.geometry.GeometryInGraph(),
                esp.mm.energy.EnergyInGraph(terms=["n2", "n3"]),
                esp.mm.energy.EnergyInGraph(terms=["n2", "n3"], suffix='_ref'),
        )

    if args.layer == "Free":
        representation = esp.nn.baselines.FreeParameterBaselineInitMean(next(iter(ds)))
        net = torch.nn.Sequential(
                representation, 
                # readout,
                # add_mean,
                esp.mm.geometry.GeometryInGraph(),
                esp.mm.energy.EnergyInGraph(terms=["n2", "n3"]),
                esp.mm.energy.EnergyInGraph(terms=["n2", "n3"], suffix='_ref'),
        )
        


    if args.metric_train == "energy":
        metrics_tr = [
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=['u', "u_ref"],
                level="g",
            ),

        ]

    elif args.metric_train == "force":
        metrics_tr = [
            esp.metrics.GraphDerivativeMetric(
                    between=["u", "u_ref"],
                    level="g",
                    weight=1.0,
                    base_metric=torch.nn.MSELoss(),
            ),
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=['u', "u_ref"],
                level="g",
            ),
        ]


    elif args.metric_train == "param":
        metrics_tr = [
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=["k", "k_ref"],
                level="n2",
            ),
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=["eq", "eq_ref"],
                level="n2",
            ),
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=["k", "k_ref"],
                level="n3",
            ),
            esp.metrics.GraphMetric(
                base_metric=torch.nn.MSELoss(),
                between=["eq", "eq_ref"],
                level="n3",
            ),
        ]

    metrics_te = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.r2,
            between=['u', 'u_ref'],
            level="g",
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.rmse,
            between=['u', 'u_ref'],
            level="g",
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.mape,
            between=['u', 'u_ref'],
            level='g',
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.mape,
            between=["k", "k_ref"],
            level="n2",
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.mape,
            between=["eq", "eq_ref"],
            level="n2",
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.mape,
            between=["k", "k_ref"],
            level="n3",
        ),
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.mape,
            between=["eq", "eq_ref"],
            level="n3",
        ),
    ]

    if args.opt == "Adam":
        opt = torch.optim.Adam(net.parameters(), 1e-5)

        if args.metric_train == "param":
            opt = torch.optim.Adam(net.parameters(), 1e-1)

    elif args.opt == "SGD":
        opt = torch.optim.SGD(net.parameters(), 1e-5, 1e-5)

    elif args.opt == "LBFGS":
        opt = torch.optim.LBFGS(net.parameters(), 1e-1, line_search_fn="strong_wolfe")

    elif args.opt == "SGLD":
        from pinot.samplers.sgld import SGLD
        opt = SGLD(net.parameters(), 1e-5)

    exp = esp.TrainAndTest(
        ds_tr=ds,
        ds_te=ds,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        normalize=esp.data.normalize.NotNormalize,
        record_interval=1,
        optimizer=opt,
        device=torch.device('cuda:0'),
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
    pickle.dump(
        exp.ref_g_test,
        open(args.out + "/g.th", 'wb'),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--out", default="results", type=str
    )
    parser.add_argument("--janossy_config", nargs="*", default=[32, "leaky_relu"])

    parser.add_argument("--n_epochs", default=10, type=int)

    parser.add_argument("--opt", default="Adam", type=str)
    parser.add_argument("--metric_train", default="energy", type=str)

    args = parser.parse_args()

    run(args)
