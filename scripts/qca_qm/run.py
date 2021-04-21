# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os
<<<<<<< HEAD
import math
=======

>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509
import numpy as np
import torch
import dgl

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

    pfizer = esp.data.dataset.GraphDataset.load('pfizer')
    coverage = esp.data.dataset.GraphDataset.load('coverage')
    emolecules = esp.data.dataset.GraphDataset.load('emolecules')
    bayer = esp.data.dataset.GraphDataset.load('bayer')
    roche = esp.data.dataset.GraphDataset.load('roche')
    fda = esp.data.dataset.GraphDataset.load('fda')
    # benchmark = esp.data.dataset.GraphDataset.load("benchmark")


    _ds_tr = pfizer + coverage + emolecules + bayer + roche
    _ds_vl = fda  # benchmark

    print(len(_ds_tr))
    print(len(_ds_vl))

    # batch
    ds_tr = _ds_tr.view("graph", batch_size=100, shuffle=False)
    ds_vl = _ds_vl.view("graph", batch_size=100)

    g = next(iter(ds_tr))
    esp.mm.geometry.geometry_in_graph(g)

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
                3: {
                    'log_coefficients': 2,
<<<<<<< HEAD
                    },
                4: {
                    'k': 6,
=======
                    'k_urey_bradley': 1,
                    'eq_urey_bradley': 1,
                    'k_bond_bond': 1,
                    'k_bond_angle': 1,
                    'k_bond_angle': 1,
                    },
                4: {
                    'k': 6,
                    'k_angle_angle': 1,
                    'k_angle_angle_torsion': 1,
                    'k_side_torsion': 1,
                    'k_center_torsion': 1,
>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509
                    },
        },
    )

    readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
        in_features=units, config=janossy_config
    )

<<<<<<< HEAD
    readout_14 = esp.nn.readout.janossy.JanossyPooling14(
        in_features=units, config=janossy_config,
    )

=======
>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509
    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
            return g
<<<<<<< HEAD

=======
>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509
    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
<<<<<<< HEAD
            # readout_14,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
    )


    torch.nn.init.normal_(
             net[1].f_out_2_to_log_coefficients.bias,
             mean=-5,
    )

    torch.nn.init.normal_(
             net[1].f_out_3_to_log_coefficients.bias,
             mean=-5,
    )

=======
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"], ii=False),
    )


    torch.nn.init.normal_(net[1].f_out_2_to_log_coefficients.bias, mean=-5,)
    torch.nn.init.normal_(net[1].f_out_3_to_log_coefficients.bias, mean=-5,)
>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509

    # net = net.cuda()
    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.center(torch.nn.MSELoss(reduction='none'), reduction="mean"),
            between=['u', "u_ref"],
            level="g",
        ),
    ]

    metrics_te = [
         esp.metrics.GraphMetric(
            base_metric=esp.metrics.center(esp.metrics.rmse),
            between=['u', "u_ref"],
            level="g",
        ),
       ]


    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_vl,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        normalize=esp.data.normalize.NotNormalize,
        record_interval=100,
        optimizer=optimizer,
        device=torch.device('cuda'),
    )

    results = exp.run()

    print(esp.app.report.markdown(results), flush=True)

    curves = esp.app.report.curve(results)

    import os
    os.mkdir(args.out)
    for spec, curve in curves.items():
        np.save(args.out + "/" + "_".join(spec) + ".npy", curve)

    net = net.cpu()

    import pandas as pd
    df = pd.DataFrame(columns=["SMILES", "RMSE", "n_snapshots"])
    import os
    torch.save(net.state_dict(), args.out + "/net.th")

<<<<<<< HEAD



    for state_name, state in exp.states.items():
        torch.save(state, args.out + "/net%s.th" % state_name)





=======
>>>>>>> b3c4c9bfb18572034beece7a41ee6dd960fbb509
    '''
    for g in _ds_tr:
        net(g.heterograph)

        _df = {
                'SMILES': g.mol.to_smiles(),
                'RMSE': 625 * esp.metrics.center(esp.metrics.rmse)(g.nodes['g'].data['u_ref'], g.nodes['g'].data['u']).cpu().detach().numpy().item(),
                'n_snapshots': g.nodes['n1'].data['xyz'].shape[1]
        }

        df = df.append(_df, ignore_index=True)

    df.to_csv(args.out + "/inspect_tr_%s.csv" % args.first)

    for g in _ds_vl:
        net(g.heterograph)

        _df = {
                'SMILES': g.mol.to_smiles(),
                'RMSE': 625 * esp.metrics.center(esp.metrics.rmse)(g.nodes['g'].data['u_ref'], g.nodes['g'].data['u']).cpu().detach().numpy().item(),
                'n_snapshots': g.nodes['n1'].data['xyz'].shape[1]
        }

        df = df.append(_df, ignore_index=True)

    df.to_csv(args.out + "/inspect_vl_%s.csv" % args.first)
    '''
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
