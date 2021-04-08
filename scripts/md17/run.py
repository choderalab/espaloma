import torch
import espaloma as esp
import copy

def run(args):
    g = esp.Graph.load(args.name)
    n_snapshot = g.nodes['g'].data['u_ref'].shape[-1]

    idxs = list(range(n_snapshot))
    import random
    random.shuffle(idxs)
    idxs_tr = idxs[:args.first]

    g_tr = copy.deepcopy(g)
    g_tr.nodes['n1'].data['xyz'] = g_tr.nodes['n1'].data['xyz'][:, idxs_tr, :]
    g_tr.nodes['g'].data['u_ref'] = g_tr.nodes['g'].data['u_ref'][:, idxs_tr]

    # layer
    layer = esp.nn.layers.dgl_legacy.gn(args.layer)

    # representation
    representation = esp.nn.Sequential(layer, config=args.config)

    # get the last bit of units
    units = [int(x) for x in args.config if isinstance(x, int)][-1]

    janossy_config = []
    for x in args.janossy_config:
        if isinstance(x, int):
            janossy_config.append(int(x))

        elif x.isdigit():
            janossy_config.append(int(x))

        else:
            janossy_config.append(x)

    print(janossy_config)

    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units, config=janossy_config,
        out_features={
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6},
        },
    )

    readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
        in_features=units, config=janossy_config
    )


    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
            return g

    net = torch.nn.Sequential(
            representation,
            readout,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4",]),
            esp.mm.energy.EnergyInGraph(suffix='_ref', terms=["n2", "n3", "n4",]),
    )


    torch.nn.init.normal_(
             net[1].f_out_2_to_log_coefficients.bias,
             mean=-5,
    )

    torch.nn.init.normal_(
             net[1].f_out_3_to_log_coefficients.bias,
             mean=-5,
    )


    metrics_tr = [
        esp.metrics.GraphMetric(
            base_metric=esp.metrics.center(torch.nn.MSELoss(reduction='none'), reduction="mean"
),
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
        ds_tr=esp.data.GraphDataset([g_tr]),
        ds_te=esp.data.GraphDataset([g]),
        net=net,
        metrics_tr=metrics_tr,
        metrics_te=metrics_te,
        n_epochs=args.n_epochs,
        normalize=esp.data.normalize.NotNormalize,
        record_interval=100,
        optimizer=optimizer,
        device=torch.device('cuda'),
    )



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol", type=str, default="benzene")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--layer", default="SAGEConv", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--config", nargs="*", default=[128, "relu", 128, "relu", 128, "relu"],
    )
    parser.add_argument(
        "--janossy_config",  nargs="*", default=[32, "leaky_relu"],
    )
    args = parser.parse_args()
    run(args)
