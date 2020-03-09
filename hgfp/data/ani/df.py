from rdkit import Chem
import h5py
import os
import hgfp

def get_ani_mol(coordinate, species, smiles):
    """ Given smiles string and list of elements as reference,
    get the RDKit mol with xyz.

    """
    # get the rdkit ref mol
    ref_mol = Chem.MolFromSmiles(smiles, sanitize=False)

    # count the number of bond types
    bond_dict = {}

    bonds = list(mol.GetBonds())

    for bond in bonds:
        bond_begin = bond.GetBeginAtom().GetSymbol()
        bond_end = bond.GetEndAtom().GetSymbol()
        bond_type = bond.GetBondType()

        if bond_dict.get((bond_begin, bond_end)) == None:
            bond_dict[(bond_begin, bond_end)] = 1
            bond_dict[(bond_end, bond_end)] = 1

        else:
            bond_dict[(bond_begin, bond_end)] += 1
            bond_dict[(bond_end, bond_end)] += 1

    # initialize a new molecule
    new_mol = Chem.RWMol(Chem.mol())


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
