# 1. Use DGL to compute atom-representations
#   (optionally also bond and graph representations)
# 2. Symmetry-pooled readout to compute atom/bond/angle/proper/improper parameters
#   (optionally also parameters for coupling terms)

from collections import namedtuple
from functools import lru_cache

import dgl
import numpy as np
import torch
from openforcefield.topology import Molecule
from sklearn.preprocessing import OneHotEncoder
from torch import nn

terms = ['atoms', 'bonds', 'angles', 'propers', 'impropers']
Readouts = namedtuple('Readouts', terms)
ParameterizedSystem = namedtuple('ParameterizedSystem', terms)


class Indices():
    def __init__(self, offmol: Molecule):
        self.atoms = np.array([a.molecule_atom_index for a in offmol.atoms])
        self.bonds = np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])
        self.angles = np.array(sorted([tuple([atom.molecule_atom_index for atom in angle]) for angle in offmol.angles]))
        self.propers = np.array(
            sorted([tuple([atom.molecule_atom_index for atom in proper]) for proper in offmol.propers]))
        self.impropers = np.array(
            sorted([tuple([atom.molecule_atom_index for atom in improper]) for improper in offmol.impropers]))


elements = [1, 3, 6, 7, 8, 9, 15, 16, 17, 19, 35, 53]
element_encoder = OneHotEncoder(sparse=False)
element_encoder.fit(np.array(elements).reshape(-1, 1))


@lru_cache(2 ** 20)
def offmol_to_dgl(offmol: Molecule) -> dgl.DGLGraph:
    graph = dgl.from_networkx(offmol.to_networkx())
    atomic_nums = [a.element.atomic_number for a in offmol.atoms]
    X = element_encoder.transform(np.array(atomic_nums).reshape(-1, 1))
    graph.ndata['element'] = torch.Tensor(X)
    return graph

@lru_cache(2**20)
def offmol_to_indices(offmol: Molecule) -> Indices:
    return Indices(offmol)


class ValenceModel(nn.Module):
    def __init__(self, node_representation: nn.Module, readouts: Readouts):
        """
        Parameters
        ----------
        node_representation is a nn.Module
            with signature node_representation.forward(graph, initial_node_reps) -> node_reps
        readouts contains nn.Modules as attributes
            so that readouts.angles(node_reps[:,1], node_reps[:,1])
        """
        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.readouts = readouts

    def forward(self, offmol: Molecule):
        indices = offmol_to_indices(offmol)
        graph = offmol_to_dgl(offmol)
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
