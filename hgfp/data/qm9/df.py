# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
from rdkit import Chem
import pandas as pd
import dgl
import torch
import os
import hgfp
import random
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openeye import oechem
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mean_and_std(csv_path='gdb9.sdf.csv'):
    df_csv = pd.read_csv(csv_path, index_col=0)
    df_u298 = df_csv['u298_atom']
    return df_u298.mean(), df_u298.std()


def topology_batched(num=-1, batch_size=16, step_size=100, sdf_path='gdb9.sdf'):
    ifs = oechem.oemolistream(sdf_path)
    mol = oechem.OEGraphMol()

    idx = 0

    while True:
        try:
            mol = next(ifs.GetOEMols())
        except:
            break

        if num != -1 and idx > num:
            break
        
        if mol != None:
            # get the name of the molecule

            g = hgfp.graph.from_oemol(mol)

            g = hgfp.heterograph.from_graph(g)

            mol = Molecule.from_openeye(mol)

            topology = Topology.from_molecules(mol)

            mol_sys = FF.create_openmm_system(topology)

            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

            simulation = Simulation(topology.to_openmm(), mol_sys, integrator)

            simulation.context.setPositions(
                0.1 * g.nodes['atom'].data['xyz'].numpy())

            xs = []

            for _ in range(batch_size):
                simulation.step(step_size)

                xs.append(
                        simulation.context.getState(
                            getPositions=True
                        ).getPositions(asNumpy=True) / angstrom)

            idx += 1

            yield g, torch.tensor(np.array(xs))



def unbatched(num=-1, sdf_path='gdb9.sdf', csv_path='gdb9.sdf.csv', hetero=False):
    # parse data
    df_csv = pd.read_csv(csv_path, index_col=0)
    df_sdf = Chem.SDMolSupplier(sdf_path)

    # get u298 only
    df_u298 = df_csv['u298_atom']

    # init
    ds = []

    idx = 0

    while True:
        try:
            mol = next(df_sdf)
        except:
            break
        if num != -1 and idx > num:
            break
        if mol != None:
            n_atoms = mol.GetNumAtoms()
            if n_atoms > 2:
                # get the name of the molecule
                name = mol.GetProp('_Name')

                # get its u298
                u = torch.squeeze(torch.Tensor([df_u298[name]]))
            
                g = hgfp.graph.from_rdkit_mol(mol)

                if hetero is True:
                    g = hgfp.heterograph.from_graph(g)

                ds.append((g, u))

                idx += 1

    random.shuffle(ds)

    return lambda: iter(ds)

def unbatched_iter(num=-1, sdf_path='gdb9.sdf', csv_path='gdb9.sdf.csv', hetero=False):
    """ Put qm9 molecules in a dataset.
    """
    # parse data
    df_csv = pd.read_csv(csv_path, index_col=0)
    df_sdf = Chem.SDMolSupplier(sdf_path)

    # get u298 only
    df_u298 = df_csv['u298_atom']

    # initialize graph list to be empty
    def qm9_iter():
        idx = 0
        while True:
            mol = next(df_sdf)
            if num != -1 and idx > num:
                break
            if mol != None:
                n_atoms = mol.GetNumAtoms()
                if n_atoms > 2:
                    # get the name of the molecule
                    name = mol.GetProp('_Name')

                    # get its u298
                    u = torch.squeeze(torch.Tensor([df_u298[name]]))

                    g = hgfp.graph.from_rdkit_mol(mol)

                    if hetero is True:
                        g = hgfp.heterograph.from_graph(g)

                    idx += 1
                    yield(g, u)



    return qm9_iter

def batched(
        num=-1,
        sdf_path='gdb9.sdf',
        csv_path='gdb9.sdf.csv',
        n_batches_in_buffer=12,
        batch_size=32,
        cache=True,
        hetero=False):

    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num, sdf_path=sdf_path, csv_path=csv_path, hetero=hetero),
        n_batches_in_buffer=n_batches_in_buffer,
        batch_size=batch_size,
        cache=cache,
        hetero=hetero)
