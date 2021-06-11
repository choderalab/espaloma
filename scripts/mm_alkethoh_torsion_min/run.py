# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os

import numpy as np
import torch

import espaloma as esp

def run(args):
    '''
    # define data
    ds_tr = esp.data.qca.pfizer()
    ds_vl = esp.data.qca.coverage()
    print(len(ds_tr))
    print(len(ds_vl))
    print("loaded", flush=True)

    # get force field
    forcefield = esp.graphs.legacy_force_field.LegacyForceField(
        args.forcefield
    )

    # param / typing
    operation = forcefield.parametrize

    # apply to dataset
    ds_tr = ds_tr.apply(operation, in_place=True)
    ds_vl = ds_vl.apply(operation, in_place=True)
    print("parametrized", flush=True)

    # apply simulation
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation
    simulation = MoleculeVacuumSimulation(
        n_samples=500, n_steps_per_sample=100,
    )
    print("simulated", flush=True)

    ds_tr = ds_tr.apply(simulation.run, in_place=True)
    ds_vl = ds_vl.apply(simulation.run, in_place=True)

    ds_tr.save('pfizer')
    ds_vl.save('coverage')
    '''

    ds = esp.data.dataset.GraphDataset.load("alkethoh")
    ds.shuffle(seed=2666)
    ds_tr, ds_vl, ds_te = ds.split([8, 1, 1])

    # batch
    ds_tr = ds_tr.view("graph", batch_size=100, shuffle=True)
    ds_vl = ds_vl.view("graph", batch_size=100, shuffle=True)
    ds_te = ds_te.view("graph", batch_size=100, shuffle=True)

    # layer
    layer = esp.nn.layers.dgl_legacy.gn(args.layer)

    # representation
    representation = esp.nn.Sequential(layer, config=args.config)

    # get the last bit of units
    units = [int(x) for x in args.config if isinstance(x, int) or (isinstance(x, str) and x.isdigit())][-1]

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
        out_features={
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6}
        },
    )

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            import math
            g.nodes['n2'].data['_coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['_coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()

            g.nodes['n2'].data['k'], g.nodes['n2'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
                g.nodes['n2'].data['_coefficients'][:, 0][:, None],
                g.nodes['n2'].data['_coefficients'][:, 1][:, None],
                1.5, 6.0
            )

            g.nodes['n3'].data['k'], g.nodes['n3'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
                g.nodes['n3'].data['_coefficients'][:, 0][:, None],
                g.nodes['n3'].data['_coefficients'][:, 1][:, None],
                0.0, math.pi,
            )

            return g


    net = torch.nn.Sequential(
            representation, 
            readout,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4"]),
            esp.mm.energy.EnergyInGraph(suffix='_ref', terms=["n2", "n3", "n4"]),
    )

    torch.nn.init.normal_(
             net[1].f_out_2_to_log_coefficients.bias,
             mean=-5,
    )

    torch.nn.init.normal_(
             net[1].f_out_3_to_log_coefficients.bias,
             mean=-5,
    )

    torch.nn.init.normal_(
            net[1].f_out_4_to_k.bias,
            std=1e-3,
    )

    torch.nn.init.normal_(
            net[1].f_out_4_to_k.weight,
            std=1e-3,
    )

    net = net.cuda()


    def rmse_min(input, target):
        return torch.nn.MSELoss()(
            input.min(dim=-1)[0],
            target.min(dim=-1)[0],
        )

    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.center(torch.nn.MSELoss(reduction='none'), reduction="mean"),
            between=['u', "u_ref"],
            level="g",
        ),

        esp.metrics.GraphMetric(
            base_metric=rmse_min,
            between=['u', 'u_ref'],
            level='g',
        ),
    ]

    metrics_te = [
         esp.metrics.GraphMetric(
            base_metric=esp.metrics.center(esp.metrics.rmse),
            between=['u', "u_ref"],
            level="g",
        ),
         esp.metrics.GraphMetric(
            base_metric=esp.metrics.rmse,
            between=['u', "u_ref"],
            level="g",
        ),
       ]


    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_te,
        ds_vl=ds_vl,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        normalize=esp.data.normalize.PositiveNotNormalize,
        record_interval=100,
        optimizer=optimizer,
        device=torch.device('cuda:0'),
    )

    results = exp.run()

    print(esp.app.report.markdown(results))


    curves = esp.app.report.curve(results)

    import os
    os.mkdir(args.out)
    for spec, curve in curves.items():
        np.save(args.out + "/" + "_".join(spec) + ".npy", curve)

    torch.save(net.state_dict(), args.out + "/net.th")

    for idx, state in exp.states.items():
        torch.save(state, args.out + "/net%s.th" % idx)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--forcefield", default="gaff-1.81", type=str)
    parser.add_argument("--layer", default="GraphConv", type=str)
    parser.add_argument("--n_classes", default=100, type=int)
    parser.add_argument(
        "--config", nargs="*", default=[128, "leaky_relu", 128, "leaky_relu", 128, "leaky_relu"],
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

    parser.add_argument("--graph_act", type=str, default="leaky_relu") 

    parser.add_argument("--n_epochs", default=10, type=int)

    parser.add_argument("--weight", default=1.0, type=float)

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=float)
    parser.add_argument("--first", default=32, type=int)
    args = parser.parse_args()

    run(args)

