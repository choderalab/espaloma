# =============================================================================
# IMPORTS
# =============================================================================
import espaloma
import abc
import torch

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(abc.ABC, torch.utils.data.Dataset):
    """ The base class of map-style dataset.

    """
    def __init__(self, graphs=None):
        super(Dataset, self).__init__()
        self.mols = mols

    def __len__(self):
        if self.graphs is None:
            return 0
        
        else:
            return len(self.mols)

    def __getitem__(self, idx):
        if self.mols is None:
            raise RuntimeError('Empty molecule dataset.')

        return self.mols[idx]

class MoleculeIterableDataset(abc.ABC, torch.utils.data.IterableDataset):
    """ The bass class of iterable-style dataset.

    """
    def __init__(self, mols):
        super(MoleculeDataset, self).__init__()
        self.mols = mols

    def __iter__(self):
        return iter(self.mols)

class MoleculeDataset(Dataset):
    """ Dataset consist of `openforcefield.topology.Molecule` objects,
    and support save and load. 

    """

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

class HomogeneousGraphDataset(Dataset):
    """ The base class of homogeneous graph dataset.

    """
    def __init__(self, graphs=None):
        super(HomogeneousGraphDataset, self).__init__()
        self.graphs = graphs

    def save(self, path):
        from dgl.data.utils import save_graphs
        save_graphs(path, graphs)

    def load(self, path):
        from dgl.data.utils import load_graphs
        # NOTE:
        # Assume no labels here
        graphs, _ = load_graphs(path)
        self.graphs = graphs

