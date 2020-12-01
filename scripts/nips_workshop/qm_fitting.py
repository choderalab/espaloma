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
from simtk.unit.quantity import Quantity


def run(args):
    
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
        u_threshold = u_min + 0.1 # hatree
        mask = torch.lt(g.nodes['g'].data['u_ref'], u_threshold).squeeze()
        
        print('%s selected' % (mask.sum().numpy().item() / mask.shape[0]))

        g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'][:, mask]

        g.nodes['n1'].data['xyz'].requires_grad = False
        g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, mask, :]
        g.nodes['n1'].data['xyz'].requires_grad = True

        g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, mask, :]


        return g

    @torch.no_grad()
    def subsample(g, n_samples=1000):
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
        [g for g in ds if g.nodes['g'].data['u_ref'].shape[1] > 1000]
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

    # ds.save('ds_lean.th')

    # ds = esp.data.dataset.GraphDataset().load('ds_lean.th')

    ds_tr, ds_te = ds.split([4, 1])

    _ds_tr = ds_tr.view(batch_size=20, shuffle=True)
    _ds_te = ds_te.view(batch_size=20, shuffle=True)

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
        [units, args.graph_act, 1024],
        [1024, args.graph_act, 1024, args.graph_act, 1],
        'u0',
    )

    class AddRef(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['k'] += g.nodes['n2'].data['k_ref']
            g.nodes['n3'].data['k'] += g.nodes['n3'].data['k_ref']
            g.nodes['n2'].data['eq'] += g.nodes['n2'].data['eq_ref']
            g.nodes['n3'].data['eq'] += g.nodes['n3'].data['eq_ref']
            return g

    add_ref = AddRef()

    net = torch.nn.Sequential(
            representation, 
            readout,
            global_readout,
            # add_ref,
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(),
            # esp.mm.energy.EnergyInGraph(suffix='_ref'),
    )
    

    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.std(torch.nn.MSELoss(reduction='none')),
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

    ]


    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    normalize = esp.data.normalize.PositiveNotNormalize

    train = esp.app.experiment.Train(
        net=net,
        data=_ds_tr,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        metrics=metrics_tr,
        normalize=normalize,
        device=torch.device('cuda:0'),
        record_interval=1000,
    )


    train.train()


    ds_te.apply(
        lambda g: subsample(g, 100),
        in_place=True,
    )


    ds_tr.apply(
        lambda g: subsample(g, 100),
        in_place=True,
    )

    ds_te = ds_te.view(batch_size=len(ds_te))
    ds_tr = ds_tr.view(batch_size=len(ds_tr))

    states = train.states

    test = esp.app.experiment.Test(
        net=net,
        data=ds_te,
        metrics=metrics_te,
        states=states,
        normalize=normalize,
    )

    test.test()

    ref_g_test = test.ref_g

    results_te = test.results

    test = esp.app.experiment.Test(
        net=net,
        data=ds_tr,
        metrics=metrics_te,
        states=states,
        normalize=normalize,
    )

    test.test()
    ref_g_training = test.ref_g
    results_tr = test.results

    results = {"test": results_te, "train": results_tr}

    print(esp.app.report.markdown(results))

    import os
    os.mkdir(args.out)

    torch.save(net.state_dict(), args.out + "/net.th")

    with open(args.out + "/result_table.md", "w") as f_handle:
        f_handle.write(esp.app.report.markdown(results))

    curves = esp.app.report.curve(results)

    for spec, curve in curves.items():
        np.save(args.out + "/" + "_".join(spec) + ".npy", curve)

    import pickle
    with open(args.out + "/ref_g_test.th", "wb") as f_handle:
        pickle.dump(ref_g_test, f_handle)

    with open(args.out + "/ref_g_training.th", "wb") as f_handle:
        pickle.dump(ref_g_training, f_handle)

    print(esp.app.report.markdown(results))


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
