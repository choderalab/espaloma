# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
import abc
import torch

# =============================================================================
# MODULE CLASSES
# =============================================================================
class MoleculeDataset(abc.ABC, torch.utils.data.Dataset):
    """ The base class of map-style dataset.

    """
    def __init__(self, mols=None):
        super(MoleculeDataset, self).__init__()
        self.mols = mols

    def __len__(self):
        if self.mols is None:
            return 0
        
        else:
            return len(self.mols)

    def __getitem__(self, idx):
        if self.mols is None:
            raise RuntimeError('Empty molecule dataset.')

        return self.mols[idx]

    def save(self, path):
        import pickle
        pickle.dump(
            [mol._g.to_dict() for mol in self.mols],
            path)

    def load(self, path):
        import pickle
        from openforcefield.topology import Molecule

        self.mols = [
            esp.MoleculeGraph(
                Molecule.from_dict(
                    _g)) for _g in pickle.load(path)]


class MoleculeIterableDataset(abc.ABC, torch.utils.data.IterableDataset):
    """ The bass class of iterable-style dataset.

    """
    def __init__(self, mols):
        super(MoleculeDataset, self).__init__()
        self.mols = mols

    def __iter__(self):
        return iter(self.mols)






    
