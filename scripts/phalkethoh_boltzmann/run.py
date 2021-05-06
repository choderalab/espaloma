# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os

import numpy as np
import torch

import espaloma as esp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer as DO
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.distributed.rpc import RRef

def run(args):
   
    dist.init_process_group(backend="mpi")
    print(dist.get_rank(), dist.get_world_size())
    import os
    path = "/data/chodera/wangyq/espaloma/scripts/data/phalkethoh_seq/_phalkethoh/"
    mols = os.listdir(path)
    mols = [mol for mol in mols if mol.isdigit()]
    import random
    random.seed(2666)
    random.shuffle(mols)
    size = int(0.8 * ( len(mols) // dist.get_world_size()))
    print(size)
    mols = mols[size*dist.get_rank():size*(dist.get_rank()+1)]
    mols = [esp.Graph.load(path + mol) for mol in mols]
    ds_tr = esp.data.dataset.GraphDataset(mols)
    ds_tr.graphs = [graph for graph in ds_tr.graphs if graph.heterograph.number_of_nodes("n4_improper") > 0]

    def fn(g):
        g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'].float()
        return g

    ds_tr.apply(fn, in_place=True)

    # batch
    ds_tr = ds_tr.view("graph", batch_size=1, shuffle=True)

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

    class GetLoss(torch.nn.Module):
        def forward(self, g):
            u = g.nodes['g'].data['u'] - g.nodes['g'].data['u'].min(dim=-1, keepdims=True)[0]
            u_ref = g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].min(dim=-1, keepdims=True)[0]
            weight = torch.softmax(
                -u_ref / (esp.units.GAS_CONSTANT * 1000),
                dim=-1,
            )
            return (weight * torch.nn.MSELoss(reduction="none")(u, u_ref)).sum()

    _net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
            GetLoss(),
    )

    net = DDP(_net.cuda(), find_unused_parameters=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for idx in range(10000):
        for g in ds_tr:
            g = g.local_var().to("cuda:0")
            optimizer.zero_grad()
            loss = net(g)
            loss.backward()
            optimizer.step()

        if dist.get_rank() == 0:
            import os
            if not os.path.exists(args.out): os.mkdir(args.out)
            torch.save(net.state_dict(), args.out + "/net%s.th" % idx)

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

