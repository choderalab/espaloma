import torch
import espaloma as esp
import copy
import numpy as np

def number(args, train_name, test_name):
    g_te = esp.Graph.load(test_name)
    n_snapshot = g_te.nodes['g'].data['u_ref'].shape[-1]

    idxs = list(range(n_snapshot))
    import random
    random.shuffle(idxs)
    idxs_te = idxs[-1000:]
    g_te.nodes['n1'].data['xyz'] = g_te.nodes['n1'].data['xyz'][:, idxs_te, :]
    g_te.nodes['g'].data['u_ref'] = g_te.nodes['g'].data['u_ref'][:, idxs_te]
 
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
            readout_improper,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper", ]),
    ).cuda()

    net.load_state_dict(
        torch.load(
            "_%s_1000_0/net.th" % train_name,
            # map_location="cpu",
        )
    )

    g_te.heterograph = g_te.heterograph.to("cuda")

    net(g_te.heterograph)

    return "%.2f" % esp.metrics.rmse(
                627.5 * (g_te.nodes['g'].data['u_ref'] - g_te.nodes['g'].data['u_ref'].mean()),
                627.5 * (g_te.nodes['g'].data['u'] - g_te.nodes['g'].data['u'].mean()),
        )


def run(args):
    mols = [
        "benzene",
        "toluene",
        "malonaldehyde",
        "salicylic",
        "aspirin",
        "ethanol",
        "uracil",
        "naphthalene",
    ]


    import pandas as pd

    df = pd.DataFrame(columns=mols)


    for mol0 in mols:
        for mol1 in mols:
            x = number(args, mol0, mol1)
            print(mol0, mol1, x)
            df[mol0][mol1] = number(args, mol0, mol1)

    print(df)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="benzene")
    parser.add_argument("--train_name", type=str, default="benzene")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--layer", default="SAGEConv", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--config", nargs="*", default=[128, "relu", 128, "relu", 128, "relu"],
    )
    parser.add_argument(
        "--janossy_config",  nargs="*", default=[128, "relu", 128, "relu", 128, "relu"],
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10,
    )
    parser.add_argument(
        "--out", type=str, default="results",
    )
    args = parser.parse_args()
    run(args)
