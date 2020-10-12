# 1. Use DGL to compute atom-representations
#   (optionally also bond and graph representations)
# 2. Symmetry-pooled readout to compute atom/bond/angle/proper/improper parameters
#   (optionally also parameters for coupling terms)

from collections import namedtuple

import dgl
import torch
from torch import nn

terms = ['atoms', 'bonds', 'angles', 'propers', 'impropers']
Indices = namedtuple('Indices', terms)
Readouts = namedtuple('Readouts', terms)
ParameterizedSystem = namedtuple('ParameterizedSystem', terms)


class ValenceModel(nn.Module):
    def __init__(self, node_representation: nn.Module, readouts: Readouts):
        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.readouts = readouts

    def forward(self, graph: dgl.DGLGraph, indices: Indices):
        node_reps = self.node_representation.forward(graph, graph.ndata['element'])

        def symmetry_pool(f, interactions, permutations):
            return sum([f(torch.cat([node_reps[interactions[:, i]] for i in perm], dim=1)) for perm in permutations])

        atoms = self.readouts.atoms(node_reps)
        bonds = symmetry_pool(self.readouts.bonds, indices.bonds, [(0, 1), (1, 0)])
        angles = symmetry_pool(self.readouts.angles, indices.angles, [(0, 1, 2), (2, 1, 0)])

        # proper torsions: sum over (abcd, dcba)
        proper_perms = [(0, 1, 2, 3), (3, 2, 1, 0)]
        propers = symmetry_pool(self.readouts.propers, indices.propers, proper_perms)

        # improper torsions: sum over (abcd, acdb, adbc)
        improper_perms = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
        impropers = symmetry_pool(self.readouts.impropers, indices.impropers, improper_perms)

        return ParameterizedSystem(atoms=atoms, bonds=bonds, angles=angles, propers=propers, impropers=impropers)
