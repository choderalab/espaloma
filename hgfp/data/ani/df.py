from rdkit import Chem
import h5py
import os
import hgfp
from openeye import oechem
import tempfile

def get_ani_mol(coordinates, species, smiles):
    """ Given smiles string and list of elements as reference,
    get the RDKit mol with xyz.

    """
    # make xyz
    fd, path = tempfile.mkstemp()

    with os.fdopen(fd, 'w') as f_handle:
        f_handle.write(str(len(species)))
        f_handle.write('\n')
        f_handle.write('\n')
        f_handle.writelines(
            ['{:8s} {:8.5f} {:8.5f} {:8.5f}'.format(
                species[idx],
                coordinates[idx][0],
                coordinates[idx][1],
                coordinates[idx][2]
            ) for idx in range(len(species))])

    # read xyz into openeye
    ifs = oechem.oemolistream()

    if ifs.open(path):
        mol = next(ifs.GetOEGraphMols())

    g = hgfp.graph.from_oemol(mol, use_fp=True)

    return g

def unbatched(ani_path='.'):
    def _iter():
        for path in os.listdir(ani_path):
            if path.endswith('.h5'):
                f = h5py.File(path, 'r')
                for d0 in list(f.keys()):
                    for d1 in list(f[d0].keys()):
                        smiles = ''.join([
                            x.decode('utf-8') for x in f[d0][d1]['smiles'].value.tolist()])
                        coordinates = f[d0][d1]['coordinates'].value
                        energies = f[d0][d1]['energies'].value
                        species = [x.decode('utf-8') for x in f[d0][d1]['species'].value]

                        low_energy_idx = np.argsort()

                        g = get_ani_mol()

                        for idx_frame in range(energies.shape[0]):

                            g.nodes['atom'].data['xyz'] = torch.Tensor(coordinates[idx_frame, :, :])
                            u = energies[idx_frame]

                            yield (g, u)
    return _iter
