from rdkit import Chem
import h5py
import os
import hgfp
import dgl
from openeye import oechem
import tempfile
import numpy as np
import torch
import copy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openeye import oechem
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')


ATOM_WEIGHT = {'H':-0.500607632585, 'C':-37.8302333826, 'N':-54.5680045287, 'O':-54.5680045287}

def get_ani_mol(coordinates, species, smiles):
    """ Given smiles string and list of elements as reference,
    get the RDKit mol with xyz.

    """

    mol = oechem.OEGraphMol()

    for symbol in species:
        mol.NewAtom(getattr(oechem, 'OEElemNo_' + symbol))

    mol.SetCoords(coordinates.reshape([-1]))
    mol.SetDimension(3)
    oechem.OEDetermineConnectivity(mol)
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEPerceiveBondOrders(mol)

    smiles_can = oechem.OECreateCanSmiString(mol)

    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)
    mol_ref = next(ims.GetOEMols())
    smiles_ref = oechem.OECreateCanSmiString(mol_ref)

    assert smiles_can == smiles_ref

    g = hgfp.graph.from_oemol(mol, use_fp=True)

    return g, mol


def mean_and_std():
    return 0.0, 1.0

def unbatched(num=-1, ani_path='.'):
    def _iter():
        idx = 0
        for path in os.listdir(ani_path):
            if idx > num and num != -1:
                break
            if path.endswith('.h5'):
                f = h5py.File(path, 'r')
                for d0 in list(f.keys()):
                    if idx > num and num != -1:
                        break
                    for d1 in list(f[d0].keys()):
                        if idx > num and num != -1:
                            break
                        try:
                            smiles = ''.join([
                                x.decode('utf-8') for x in f[d0][d1]['smiles'].value.tolist()])
                            coordinates = f[d0][d1]['coordinates'].value
                            energies = f[d0][d1]['energies'].value
                            species = [x.decode('utf-8') for x in f[d0][d1]['species'].value]

                            low_energy_idx = np.argsort(energies)[0]

                            g, mol = get_ani_mol(
                                        coordinates[low_energy_idx],
                                        species,
                                        smiles)

                            g = hgfp.heterograph.from_graph(g)


                            u0 = np.sum([ATOM_WEIGHT[x] for x in species])
                            u_min = energies[low_energy_idx] - u0

                            for idx_frame in range(energies.shape[0]):
                                u = energies[idx_frame] - u0

                                if u < u_min + 0.44:
                                    g.nodes['atom'].data['xyz'] = torch.Tensor(coordinates[idx_frame, :, :])
                                    u = torch.squeeze(torch.Tensor([u]))
                                    idx += 1

                                    yield (g, u)

                        except:
                            continue

    return _iter

def topology_batched_md(num=-1, batch_size=16, step_size=100, ani_path='.'):

    idx = 0
    for path in os.listdir(ani_path):
        if idx > num and num != -1:
            break
        if path.endswith('.h5'):
            f = h5py.File(path, 'r')
            for d0 in list(f.keys()):
                if idx > num and num != -1:
                    break
                for d1 in list(f[d0].keys()):


                    if idx > num and num != -1:
                        break
                    
                    try:
                        smiles = ''.join([
                            x.decode('utf-8') for x in f[d0][d1]['smiles'].value.tolist()])
                        coordinates = f[d0][d1]['coordinates'].value
                        energies = f[d0][d1]['energies'].value
                        species = [x.decode('utf-8') for x in f[d0][d1]['species'].value]

                        low_energy_idx = np.argsort(energies)[0]

                        g, mol = get_ani_mol(
                                    coordinates[low_energy_idx],
                                    species,
                                    smiles)

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
                                    ).getPositions(asNumpy=False) / angstrom)

                        idx += 1

                        yield g, torch.tensor(xs)
                    
                    
                    except:
                        pass

def topology_batched(num=-1, ani_path='.', mm=False, batch_size=64, *args, **kwargs):
    class _iter():
        def __iter__(self):
            idx = 0
            for path in os.listdir(ani_path):
                if idx > num and num != -1:
                    break
                if path.endswith('.h5'):
                    f = h5py.File(path, 'r')
                    for d0 in list(f.keys()):
                        if idx > num and num != -1:
                            break
                        for d1 in list(f[d0].keys()):
                            if idx > num and num != -1:
                                break
                            smiles = ''.join([
                                x.decode('utf-8') for x in f[d0][d1]['smiles'].value.tolist()])
                            coordinates = f[d0][d1]['coordinates'].value
                            energies = f[d0][d1]['energies'].value
                            species = [x.decode('utf-8') for x in f[d0][d1]['species'].value]
                            

                            low_energy_idx = np.argsort(energies)[0]
                            
                            try:
                                g, mol = get_ani_mol(
                                            coordinates[low_energy_idx],
                                            species,
                                            smiles)
                               
                            except:
                                continue

                            g_h = hgfp.hierarchical_graph.from_oemol(mol)

                            g = hgfp.heterograph.from_graph(g)

                            u0 = np.sum([ATOM_WEIGHT[x] for x in species])
                            u_min = energies[low_energy_idx] - u0

                            try:
                                    g = hgfp.data.mm_energy.u(mol, toolkit='openeye', return_graph=True)
                            except:
                                    continue


                            import random
                            idxs = np.array(random.choices(
                                    list(range(energies.shape[0])),
                                    k=batch_size))
                            
                            idxs = idxs[energies[idxs] < u_min + 0.1]
                            
                            us = torch.tensor(energies[idxs])
                            xs = torch.tensor(coordinates[idxs])

                            idx += 1
                            yield g, xs, us, g_h
                    

    return _iter()

def batched(
        num=-1,
        n_batches_in_buffer=12,
        batch_size=32,
        cache=True,
        hetero=True):

    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num),
        n_batches_in_buffer=n_batches_in_buffer,
        batch_size=batch_size,
        cache=cache,
        hetero=hetero)
