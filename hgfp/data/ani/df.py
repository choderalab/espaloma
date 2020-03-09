from rdkit import Chem
import h5py
import os
import hgfp

def unbatched(num=-1, ani_path=os.getcwd()):
    print(ani_path)
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

                        mol = Chem.MolFromSmiles(smiles)
                        mol = Chem.AddHs(mol)

                        print([atom.GetSymbol() for atom in mol.GetAtoms()])
                        print(species)
                        assert [atom.GetSymbol() for atom in mol.GetAtoms()] == species

                        g = hgfp.heterograph.from_graph(
                            hgfp.graph.from_rdkit_mol(
                                mol))

                        for idx_frame in range(energies.shape[0]):

                            g.nodes['atom'].data['xyz'] = torch.Tensor(coordinates[idx_frame, :, :])
                            u = energies[idx_frame]

                            yield (g, u)
    return _iter
