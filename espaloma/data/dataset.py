# =============================================================================
# IMPORTS
# =============================================================================
import abc

import torch

import espaloma as esp


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(abc.ABC, torch.utils.data.Dataset):
    """The base class of map-style dataset.

    Parameters
    ----------
    graphs : List
        objects in the dataset

    Methods
    -------
    shuffle
        Randomly shuffle the graphs in the dataset.

    apply(fn, in_place=True)
        Apply a function to every graph in the dataset.
        If `in_place=True`, modify the graph in-place.

    split(partitions)
        Split the dataset into partitions

    subsample(ratio, seed=None)
        Subsample the dataset.

    save(path)
        Save the dataset to a local path.

    load(path)
        Load a dataset from local path.

    Note
    ----
    This also supports iterative-style dataset by deleting `__getitem__`
    and `__len__` function.

    Attributes
    ----------
    transforms : an iterable of callables that transforms the input.
        the `__getiem__` method applies these transforms later.

    Examples
    --------
    >>> data = Dataset([esp.Graph("C")])

    """

    def __init__(self, graphs=None):
        super(Dataset, self).__init__()
        self.graphs = graphs
        self.transforms = None

    def __len__(self):
        # 0 len if no graphs
        if self.graphs is None:
            return 0

        else:
            return len(self.graphs)

    def __getitem__(self, idx):
        if self.graphs is None:
            raise RuntimeError("Empty molecule dataset.")

        if isinstance(idx, int):  # sinlge element
            if self.transforms is None:  # when no transform act like list
                return self.graphs[idx]

            else:
                graph = self.graphs[idx]

                # nested transforms
                for transform in self.transforms:
                    graph = transform(graph)

                return graph

        elif isinstance(idx, slice):
            # implement slicing
            if self.transforms is None:
                # return a Dataset object rather than list
                return self.__class__(graphs=self.graphs[idx])
            else:
                graphs = []
                for graph in self.graphs[idx]:

                    # nested transforms
                    for transform in self.transforms:
                        graph = transform(graph)
                    graphs.append(graph)

                return self.__class__(graphs=graphs)

        elif isinstance(idx, list):
            # implement slicing
            if self.transforms is None:
                # return a Dataset object rather than list
                return self.__class__(
                    graphs=[self.graphs[_idx] for _idx in idx]
                )
            else:
                graphs = []
                for _idx in idx:
                    graph = self[_idx]
                    # nested transforms
                    for transform in self.transforms:
                        graph = transform(graph)
                    graphs.append(graph)

                return self.__class__(graphs=graphs)

    def __iter__(self):
        if self.transforms is None:
            return iter(self.graphs)

        else:
            # TODO:
            # is this efficient?
            graphs = iter(self.graphs)
            for transform in self.transforms:
                graphs = map(transform, graphs)

            return graphs

    def shuffle(self, seed=None):
        import random
        from random import shuffle

        if seed is not None:
            random.seed(seed)

        shuffle(self.graphs)
        return self

    def apply(self, fn, in_place=False):
        r"""Apply functions to the elements of the dataset.

        Parameters
        ----------
        fn : callable

        Note
        ----
        If in_place is False, `fn` is added to the `transforms` else it is applied
        to elements and modifies them.

        """
        assert callable(fn)
        assert isinstance(in_place, bool)

        if in_place is False:  # add to list of transforms
            if self.transforms is None:
                self.transforms = []

            self.transforms.append(fn)

        else:  # modify in-place
            # self.graphs = list(map(fn, self.graphs))
            _graphs = []
            for graph in self.graphs:
                try:
                    _graphs.append(fn(graph))
                except:
                    pass
            self.graphs = _graphs

        return self  # to allow grammar: ds = ds.apply(...)

    def split(self, partition):
        """Split the dataset according to some partition.

        Parameters
        ----------
        partition : sequence of integers or floats

        """
        n_data = len(self)
        p_sizes = []
        for i, _partition in enumerate(partition):
            p_size = int((n_data - sum(p_sizes)) * _partition / sum(partition[i:]))
            p_sizes.append(p_size)
        assert sum(p_sizes) == n_data, f"{p_sizes}, {sum(p_sizes)}"
        ds = []
        idx = 0
        for p_size in p_sizes:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds

    def subsample(self, ratio, seed=None):
        """Subsample the dataset according to some ratio.

        Parameters
        ----------
        ratio : float
            Ratio between the size of the subsampled dataset and the
            original dataset.

        """
        n_data = len(self)
        idxs = list(range(n_data))
        import random

        random.seed(seed)
        _idxs = random.choices(idxs, k=int(n_data * ratio))
        return self[_idxs]

    def save(self, path):
        """Save dataset to path.

        Parameters
        ----------
        path : path-like object
        """
        import pickle

        with open(path, "wb") as f_handle:
            pickle.dump(self.graphs, f_handle)

    def regenerate_impropers(self, improper_def="smirnoff"):
        """
        Regenerate the improper nodes for all graphs.

        Parameters
        ----------
        improper_def : str
            Which convention to use for permuting impropers.
        """
        from espaloma.graphs.utils.regenerate_impropers import (
            regenerate_impropers,
        )

        for g in self.graphs:
            regenerate_impropers(g, improper_def)

    @classmethod
    def load(cls, path):
        """Load path to dataset.

        Parameters
        ----------
        """
        import pickle

        with open(path, "rb") as f_handle:
            graphs = pickle.load(f_handle)

        return cls(graphs)

    def __add__(self, x):
        return self.__class__(self.graphs + x.graphs)


class GraphDataset(Dataset):
    """Dataset with additional support for only viewing
    certain attributes as `torch.utils.data.DataLoader`

    Methods
    -------
    view(collate_fn, *args, **kwargs)
        Provide a `torch.utils.data.DataLoader` view of the dataset.

    Note
    """

    def __init__(self, graphs=[], first=None):
        super(GraphDataset, self).__init__()
        from openff.toolkit.topology import Molecule

        if all(
            isinstance(graph, Molecule) or isinstance(graph, str)
            for graph in graphs
        ):

            if first is None or first == -1:
                graphs = [esp.Graph(graph) for graph in graphs]

            else:
                graphs = [esp.Graph(graph) for graph in graphs[:first]]

        self.graphs = graphs

    @staticmethod
    def batch(graphs):
        import dgl

        if all(isinstance(graph, esp.graphs.graph.Graph) for graph in graphs):
            return dgl.batch([graph.heterograph for graph in graphs])

        elif all(isinstance(graph, dgl.DGLGraph) for graph in graphs):
            return dgl.batch(graphs)

        elif all(isinstance(graph, dgl.DGLHeteroGraph) for graph in graphs):
            return dgl.batch(graphs)

        else:
            raise RuntimeError(
                "Can only batch DGLGraph or DGLHeterograph,"
                "now have %s" % type(graphs[0])
            )

    def view(self, collate_fn="graph", *args, **kwargs):
        """Provide a data loader.

        Parameters
        ----------
        collate_fn : callable or string
            see `collate_fn` argument for `torch.utils.data.DataLoader`


        """
        if collate_fn == "graph":
            collate_fn = self.batch

        elif collate_fn == "homograph":

            def collate_fn(graphs):
                graph = self.batch([g.homograph for g in graphs])

                return graph

        elif collate_fn == "graph-typing":

            def collate_fn(graphs):
                graph = self.batch(graphs)
                y = graph.ndata["legacy_typing"]
                return graph, y

        elif collate_fn == "graph-typing-loss":
            loss_fn = torch.nn.CrossEntropyLoss()

            def collate_fn(graphs):
                graph = self.batch(graphs)
                loss = lambda _graph: loss_fn(
                    _graph.ndata["nn_typing"], graph.ndata["legacy_typing"]
                )
                return graph, loss

        return torch.utils.data.DataLoader(
            dataset=self, collate_fn=collate_fn, *args, **kwargs
        )

    def save(self, path):
        import os

        os.mkdir(path)
        for idx, graph in enumerate(self.graphs):
            graph.save(path + "/" + str(idx))

    @classmethod
    def load(cls, path):
        import os

        paths = os.listdir(path)
        paths = [_path for _path in paths]

        graphs = []
        for _path in paths:
            graphs.append(esp.Graph.load(path + "/" + _path))

        return cls(graphs)
