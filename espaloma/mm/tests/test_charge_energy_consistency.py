import pytest
import espaloma as esp
import numpy as np
import numpy.testing as npt
import pytest
import torch
from simtk import openmm
from simtk import openmm as mm
from simtk import unit

@pytest.mark.parametrize(
    "g",
    esp.data.esol(first=1), # use a subset of ESOL dataset to test
    # [esp.Graph("c1ccccc1")],
)
def test_coulomb_energy_consistency(g):
    """ We use both `esp.mm` and OpenMM to compute the Coulomb energy of
    some molecules with generated geometries and see if the resulting Columb
    energy matches.


    """
    # make simulation
    from espaloma.data.md import MoleculeVacuumSimulation

    print(g.mol)

    # get simulation
    esp_simulation = MoleculeVacuumSimulation(
        n_samples=1,
        n_steps_per_sample=10,
        forcefield="gaff-1.81",
        charge_method="gasteiger",
    )

    simulation = esp_simulation.simulation_from_graph(g)
    charges = g.mol.partial_charges.flatten()
    system = simulation.system

    esp_simulation.run(g, in_place=True)

    # if MD blows up, forget about it
    if g.nodes["n1"].data["xyz"].abs().max() > 100:
        return True

    g.nodes['n1'].data['q'] = torch.tensor(charges).unsqueeze(-1)
    esp.mm.nonbonded.multiply_charges(g.heterograph)
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, terms=["nonbonded", "onefour"])
    g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u'].clone()
    esp.data.md.subtract_coulomb_force(g)



    npt.assert_almost_equal(
        g.nodes['g'].data['u_ref'].item(),
        0.0,
        decimal=3,
    )
