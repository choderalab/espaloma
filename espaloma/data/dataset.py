# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
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
        self.graphs = graphs

    def __len__(self):
        if self.graphs is None:
            return 0
        
        else:
            return len(self.graphs)

    def __getitem__(self, idx):
        if self.graphs is None:
            raise RuntimeError('Empty molecule dataset.')
        
        if isinstance(idx, int): # sinlge element
            return self.graphs[idx]
        elif isinstance(idx, slice): # implement slicing
            import copy
            ds = copy.copy(self)
            ds.graphs = self.graphs[idx]
            return ds

    def __iter__(self):
        return iter(self.graphs)

    def apply(self, fn):
        """ Apply function to all elements in the dataset.

        Note
        ----
        This function modifies in-place and also return the list.
        """
        graphs = [fn(graph) for graph in self.graphs]
        self.graphs = graphs
        return graphs


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


    def to_homogeneous_with_legacy_typing(self, forcefield='gaff-1.81'):
        """ Apply legacy atom typing scheme to yield a
        `HomogeneousGraphDataset`.

        Parameters
        ----------
        forcefield : str or `espaloma.graphs.LegacyForceField.
            
        """
        assert isinstance(forcefield, str) or isinstance(
                forcefield,
                esp.graphs.LegacyForceField
            ), 'has to be either force field object or string'

        if isinstance(forcefield, str):
            forcefield = esp.graphs.LegacyForceField(forcefield)

        return HomogeneousGraphDataset(self.apply(forcefield.typing))


class HomogeneousGraphDataset(Dataset):
    """ The base class of homogeneous graph dataset.

    """
    def __init__(self, graphs=None):
        super(HomogeneousGraphDataset, self).__init__()
        self.graphs = graphs

    def save(self, path):
        from dgl.data.utils import save_graphs
        save_graphs(path, self.graphs)

    def load(self, path):
        from dgl.data.utils import load_graphs
        # NOTE:
        # Assume no labels here
        graphs, _ = load_graphs(path)
        self.graphs = graphs


class HeterogeneousGraphDataset(Dataset):
    """ The base class of heterogeneous graph dataset.

    """
    def __init__(self, graphs=None):
        super(HeterogeneousGraphDataset, self).__init__()
        self.graphs = graphs

    def save(self, path):
        import dgl
        from dgl.data.utils import save_graphs
        graphs = [dgl.to_homo(graph._g) for graph in self.graphs]

        # TODO:
        # this doesn't need to be all the same
        assert all(graphs[0]._g.ntype == graph._g.ntype for graph in graphs)
        assert all(graphs[0]._g.etype == graph._g.etype for graph in graphs)

        labels = {'ntype': graphs[0]._g.ntype, 'etype': graphs[0]._g.etype}
        save_graphs(path, graphs, labels)

    def load(self, path):
        import dgl
        from dgl.data.utils import load_graphs
        graphs, labels = load_graphs(path)
        graphs = [dgl.to_hetero(
            graph,
            ntypes=labels['ntype'],
            etypes=labels['etype']
        )]

        self.graphs = graphs
