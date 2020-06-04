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

    Parameters
    ----------
    graphs : list
        objects in the dataset

    Note
    ----
    This also supports iterative-style dataset by deleting `__getitem__`
    and `__len__` function. 

    Attributes
    ----------
    transforms : an iterable of callables that transforms the input.

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
            raise RuntimeError('Empty molecule dataset.')

        if isinstance(idx, int): # sinlge element
            if self.transforms is None: # when no transform act like list
                return self.graphs[idx]

            else:
                graph = self.graphs[idx]

                # nested transforms
                for transform in self.transforms:
                    graph = transform(graph)

                return graph

        elif isinstance(idx, slice): # implement slicing
            if self.transforms is None:
                # return a Dataset object rather than list
                return self.__class__(graphs=self.graphs[idx]) 
            else:
                graphs = []
                for graph in self.graphs:

                    # nested transforms
                    for transform in transforms:
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
                graphs = map(
                        transform,
                        graphs)

            return graphs

    def apply(self, fn, in_place=False):
        assert callable(fn)
        assert isinstance(in_place, bool)

        if in_place is False: # add to list of transforms
            if self.transforms is None:
                self.transforms = []

            self.transforms.append(fn)

        else: # modify in-place
            self.graphs = list(map(fn, self.graphs))

        return self # to allow grammar: ds = ds.apply(...)

    def split(self, partition):
        """ Split the dataset according to some partition.

        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        ds = []
        idx = 0
        for p_size in partition:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds

class GraphDataset(Dataset):
    """ Dataset with additional support for only viewing
    certain attributes as `torch.utils.data.DataLoader`

    
    """

    def __init__(self, graphs, first=None):
        super(GraphDataset, self).__init__()
        from openforcefield.topology import Molecule

        if all(
                isinstance(
                    graph, 
                    Molecule
                ) or isinstance(
                    graph,
                    str) for graph in graphs):

            if first is None:
                graphs = [esp.Graph(graph) for graph in graphs]
            else:
                graphs = [esp.Graph(graph) for graph in graphs[:first]]

        self.graphs = graphs

    @staticmethod
    def batch(graphs):
        import dgl
        if all(isinstance(graph, esp.graphs.graph.Graph) for graph in graphs):
            return dgl.batch([graph.homograph for graph in graphs])

        elif all(isinstance(graph, dgl.DGLGraph) for graph in graphs):
            return dgl.batch(self.graphs)
        
        elif all(isinstance(graph, dgl.DGLHeteroGraph) for graph in graphs):
            return dgl.batch_hetero(self.graphs)

        else:
            raise RuntimeError('Can only batch DGLGraph or DGLHeterograph,'
                'now have %s' % type(graphs[0]))

    def view(self, collate_fn='graph', *args, **kwargs):
        if collate_fn == 'graph':
            collate_fn = self.batch
        
        elif collate_fn == 'graph-typing':
            def collate_fn(graphs):
                graph = self.batch(graphs)
                y = graph.ndata['legacy_typing']
                return graph, y

        elif collate_fn == 'graph-typing-loss':
            loss_fn = torch.nn.CrossEntropyLoss()
            def collate_fn(graphs):
                graph = self.batch(graphs)
                loss = lambda _graph: loss_fn(
                        _graph.ndata['nn_typing'],
                        graph.ndata['legacy_typing'])
                return graph, loss

        return torch.utils.data.DataLoader(
                dataset=self,
                collate_fn=collate_fn,
                *args, **kwargs)




