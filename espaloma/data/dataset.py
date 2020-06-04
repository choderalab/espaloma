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
        self.transforms = None

    def __len__(self):
        if self.graphs is None:
            return 0
        
        else:
            return len(self.graphs)

    def __getitem__(self, idx):
        if self.graphs is None:
            raise RuntimeError('Empty molecule dataset.')

        if isinstance(idx, int): # sinlge element
            if self.transforms is None:
                return self.graphs[idx]
            else:
                graph = self.graphs[idx]
                for transform in self.transforms:
                    graph = transform(graph)

                return graph

        elif isinstance(idx, slice): # implement slicing
            if self.transforms is None:
                return Dataset(graphs=self.graphs[idx])
            else:
                graphs = []
                for graph in self.graphs:
                    for transform in transforms:
                        graph = transform(graph)
                    graphs.append(graph)

                return Dataset(graphs=graphs)

    def __iter__(self):
        if self.transforms is None:
            return iter(self.graphs)

        else:
            graphs = iter(self.graphs)
            for transform in self.transforms:
                graphs = map(
                        transform,
                        graphs)

            return graphs

    def apply(self, fn, in_place=False):
        assert callable(fn)
        assert isinstance(in_place, bool)

        if in_place is False:
            if self.transforms is None:
                self.transforms = []

            self.transforms.append(fn)

        else:
            self.graphs = list(map(fn, self.graphs))

        return self


class GraphDataset(Dataset):
    """ Batch dataset with additional support for only viewing
    certain attributes.

    """

    def __init__(self, graphs):
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    @staticmethod
    def batch(graphs):
        import dgl
        if all(isinstance(graph, dgl.DGLGraph) for graph in self.graphs):
            return dgl.batch(self.graphs)
        
        elif all(isinstance(graph, dgl.DGLHeterograph) for graph in graphs):
            return dgl.batch_hetero(self.graphs)

        else:
            raise RuntimeError('Can only batch DGLGraph or DGLHeterograph')

    

