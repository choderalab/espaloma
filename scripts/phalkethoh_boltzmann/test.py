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

def run():
    path = "/data/chodera/wangyq/espaloma/scripts/data/phalkethoh_seq/_phalkethoh/"
    mols = os.listdir(path)
    mols = [esp.Graph.load(path+mol) for mol in mols if mol.isdigit()]
    chunk_size = int(0.1 * len(mols))
    ds_vl = esp.data.dataset.GraphDataset(mols[8*chunk_size:9*chunk_size])
    ds_te = esp.data.dataset.GraphDataset(mols[9*chunk_size:])

    # layer
    layer = esp.nn.layers.dgl_legacy.gn("SAGEConv")

    # representation
    representation = esp.nn.Sequential(layer, config=[128, "relu", 128, "relu", 128, "relu"])
    janossy_config = [128, "relu", 128, "relu", 128, "relu"]
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=128, config=janossy_config,
        out_features={
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6},
        },
    )

    readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
        in_features=128, config=janossy_config
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
                -u_ref / (esp.units.GAS_CONSTANT * 300),
                dim=-1,
            )
            return (weight * torch.nn.MSELoss(reduction="none")(u, u_ref)).sum().pow(0.5)

    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
            # GetLoss(),
    )

    
    state_dict = torch.load(
            "128_SAGEConv_relu_1.0_1e-4_1___single_gpu_janossy_first_distributed_warm/net79.th",
            map_location=torch.device("cpu"),
    )

    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)

    u = []
    u_ref = []

    for g in ds_vl:
        net(g.heterograph)
        u.append(g.nodes['g'].data['u'].detach().numpy())
        u_ref.append(g.nodes['g'].data['u_ref'].detach().numpy())

    np.savez("u", *u)
    np.savez("u_ref", *u_ref)

if __name__ == "__main__":
    run()
