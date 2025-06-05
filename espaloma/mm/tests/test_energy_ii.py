import pytest
import espaloma as esp
import torch
import dgl


def test_energy():
    g = esp.Graph("c1ccccc1")

    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    simulation = MoleculeVacuumSimulation(n_samples=10, n_steps_per_sample=10)
    g = simulation.run(g, in_place=True)

    param = esp.graphs.legacy_force_field.LegacyForceField(
        "gaff-1.81"
    ).parametrize

    g = param(g)

    # parametrize

    # layer
    layer = esp.nn.layers.dgl_legacy.gn()

    # representation
    representation = esp.nn.Sequential(
        layer, config=[32, "relu", 32, "relu", 32, "relu"]
    )

    # get the last bit of units
    units = 32

    janossy_config = [32, "relu"]

    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units,
        config=janossy_config,
        out_features={
            2: {"log_coefficients": 2},
            3: {
                "log_coefficients": 2,
                "coefficients_urey_bradley": 2,
                "k_bond_bond": 1,
                "k_bond_angle": 1,
                "k_bond_angle": 1,
            },
            4: {
                "k": 6,
                "k_angle_angle": 1,
                "k_angle_angle_torsion": 1,
                "k_angle_torsion": 1,
                "k_side_torsion": 1,
                "k_center_torsion": 1,
            },
        },
    )

    readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
        in_features=units, config=janossy_config
    )

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes["n2"].data["coefficients"] = (
                g.nodes["n2"].data["log_coefficients"].exp()
            )
            g.nodes["n3"].data["coefficients"] = (
                g.nodes["n3"].data["log_coefficients"].exp()
            )
            return g

    class CarryII(torch.nn.Module):
        def forward(self, g):
            import math

            g.multi_update_all(
                {
                    "n2_as_0_in_n3": (
                        dgl.function.copy_u("u", "m_u_0"),
                        dgl.function.sum("m_u_0", "u_left"),
                    ),
                    "n2_as_1_in_n3": (
                        dgl.function.copy_u("u", "m_u_1"),
                        dgl.function.sum("m_u_1", "u_right"),
                    ),
                    "n2_as_0_in_n4": (
                        dgl.function.copy_u("u", "m_u_0"),
                        dgl.function.sum("m_u_0", "u_bond_left"),
                    ),
                    "n2_as_1_in_n4": (
                        dgl.function.copy_u("u", "m_u_1"),
                        dgl.function.sum("m_u_1", "u_bond_center"),
                    ),
                    "n2_as_2_in_n4": (
                        dgl.function.copy_u("u", "m_u_2"),
                        dgl.function.sum("m_u_2", "u_bond_right"),
                    ),
                    "n3_as_0_in_n4": (
                        dgl.function.copy_u("u", "m3_u_0"),
                        dgl.function.sum("m3_u_0", "u_angle_left"),
                    ),
                    "n3_as_1_in_n4": (
                        dgl.function.copy_u("u", "m3_u_1"),
                        dgl.function.sum("m3_u_1", "u_angle_right"),
                    ),
                },
                cross_reducer="sum",
            )

            return g

    net = torch.nn.Sequential(
        representation,
        readout,
        readout_improper,
        ExpCoeff(),
        esp.mm.geometry.GeometryInGraph(),
        esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
        CarryII(),
        esp.mm.energy.EnergyInGraphII(),
    )

    torch.nn.init.normal_(
        net[1].f_out_2_to_log_coefficients.bias,
        mean=-5,
    )
    torch.nn.init.normal_(
        net[1].f_out_3_to_log_coefficients.bias,
        mean=-5,
    )

    for name, module in net[1].named_modules():
        if "k" in name:
            torch.nn.init.normal(module.bias, mean=0.0, std=1e-4)
            torch.nn.init.normal(module.weight, mean=0.0, std=1e-4)

    g = net(g.heterograph)

    print(g.nodes["n3"].data)
    print(g.nodes["n4"].data)

    # print(g.nodes['n2'].data)
    esp.mm.geometry.geometry_in_graph(g)
    esp.mm.energy.energy_in_graph(g)
