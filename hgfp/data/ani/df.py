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


def topology_batched(num=-1, ani_path='.', mm=False, *args, **kwargs):
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

                            
                            g = hgfp.heterograph.from_graph(g)

                            u0 = np.sum([ATOM_WEIGHT[x] for x in species])
                            u_min = energies[low_energy_idx] - u0

                            if mm == True:
                                try:
                                    g_ = hgfp.data.mm_energy.u(mol, toolkit='openeye', return_graph=True)
                                except:
                                    continue

                            gs = []
                            us = []
                            
                            for idx_frame in range(energies.shape[0]):
                                u = energies[idx_frame] - u0
                                if u < u_min + 0.44:
                                    g = copy.deepcopy(g_)
                                    g.apply_nodes(lambda node: {
                                        'xyz': torch.Tensor(coordinates[idx_frame, :, :])},
                                        ntype='atom')

                                    if mm == True:

                                        g = hgfp.mm.geometry_in_heterograph.from_heterograph_with_xyz(g)

                                        g = hgfp.mm.energy_in_heterograph.u(g)

                                        u = torch.sum(
                                                torch.cat(
                                                [
                                                    g.nodes['mol'].data['u' + term][:, None] for term in [
                                                        'bond'#, 'angle', 'torsion', 'one_four', 'nonbonded', '0'
                                                ] if 'u' + term in g.nodes['mol'].data],
                                                dim=1),
                                            dim=1)

                                       

                                    gs.append(g)
                                    us.append(u)
                                
                            gs = dgl.batch_hetero(gs)
                            us = torch.stack(us)

                            idx += 1
                            yield gs, us
                    

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
