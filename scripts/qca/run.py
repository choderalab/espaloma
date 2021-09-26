# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

import espaloma as esp
from simtk import unit
from simtk.unit import Quantity


def run(args):

    '''
    ds = esp.data.dataset.GraphDataset().load(
        'ds.th',
        )

    def subtract_offset(g):
        elements = [atom.atomic_number for atom in g.mol.atoms]
        offset = esp.data.utils.sum_offsets(elements)
        g.nodes['g'].data['u_ref'] -= offset
        return g

    @torch.no_grad()
    def exclude_high_energy(g):
        u_min = g.nodes['g'].data['u_ref'].min()
        u_threshold = u_min + 0.01 # hatree
        mask = torch.lt(g.nodes['g'].data['u_ref'], u_threshold).squeeze()

        print('%s selected' % (mask.sum().numpy().item() / mask.shape[0]))

        g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'][:, mask]

        g.nodes['n1'].data['xyz'].requires_grad = False
        g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, mask, :]
        g.nodes['n1'].data['xyz'].requires_grad = True

        g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, mask, :]


        return g

    @torch.no_grad()
    def subsample(g, n_samples=100):
        n_total_samples = g.nodes['g'].data['u_ref'].shape[1]
        mask = np.random.choice(list(range(n_total_samples)), n_samples, replace=False).tolist()

        g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'][:, mask]

        g.nodes['n1'].data['xyz'].requires_grad = False
        g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, mask, :]
        g.nodes['n1'].data['xyz'].requires_grad = True

        g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, mask, :]

        return g
    ds.apply(
        subtract_offset,
        in_place=True,
    )

    ds.apply(
        exclude_high_energy,
        in_place=True,
    )

    ds = esp.data.dataset.GraphDataset(
        [g for g in ds if g.nodes['g'].data['u_ref'].shape[1] > 100]
    )

    print(ds.graphs)

    ds.apply(
        subsample,
        in_place=True
    )

    print(len(ds))

    ds.apply(
        esp.data.md.subtract_nonbonded_force,
        in_place=True,
    )

    ds.save('ds_lean.th')
    '''

    ds = esp.data.dataset.GraphDataset().load('ds_lean.th')[:100]

    ds_tr, ds_te = ds.split([4, 1])

    ds_tr = ds_tr.view(batch_size=80, shuffle=True)
    ds_te = ds_te.view(batch_size=20, shuffle=True)

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
            1: {'sigma': 1, 'epsilon': 1},
            2: {'k': 1, 'eq': 1},
            3: {'k': 1, 'eq': 1},
            4: {'k': 6},
        },
    )

    global_readout = esp.nn.readout.graph_level_readout.GraphLevelReadout(
        units,
        [units, args.graph_act],
        [units, args.graph_act, 1],
        'u0',
    )

    net = torch.nn.Sequential(
            representation,
            readout,
            global_readout,
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(),
            # esp.mm.energy.EnergyInGraph(suffix='_ref'),
    )


    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(),
            between=['u', "u_ref"],
            level="g",
        ),

        esp.metrics.GraphHalfDerivativeMetric(
               base_metric=torch.nn.MSELoss(),
               weight=args.weight,
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
        esp.metrics.GraphHalfDerivativeMetric(
              base_metric=esp.metrics.r2,
              weight=1.0,
        ),
        esp.metrics.GraphHalfDerivativeMetric(
              base_metric=esp.metrics.rmse,
              weight=1.0,
        ),


    ]


    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_te=ds_te,
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        record_interval=1000,
        normalize=esp.data.normalize.PositiveNotNormalize,
        optimizer=lambda net: torch.optim.Adam(net.parameters(), args.lr),
        device=torch.device('cuda:0'),
    )

    results = exp.run()

    print(esp.app.report.markdown(results))

    import os
    os.mkdir(args.out)

    torch.save(net.state_dict(), args.out + "/net.th")

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

    parser.add_argument("--graph_act", type=str, default="leaky_relu")

    parser.add_argument("--n_epochs", default=10, type=int)

    parser.add_argument("--weight", default=1.0, type=float)

    parser.add_argument("--lr", default=1e-5, type=float)

    args = parser.parse_args()

    run(args)
