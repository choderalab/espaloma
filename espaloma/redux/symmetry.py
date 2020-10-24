# 1. Use DGL to compute atom-representations
#   (optionally also bond and graph representations)
# 2. Symmetry-pooled readout to compute atom/bond/angle/proper/improper params
#   (optionally also parameters for coupling terms)

from collections import namedtuple
from functools import lru_cache

import dgl
import numpy as np
import torch
from openforcefield.topology import Molecule
from sklearn.preprocessing import OneHotEncoder
from torch import nn

from espaloma.graphs.utils import offmol_indices

terms = ["atoms", "bonds", "angles", "propers", "impropers"]
Readouts = namedtuple("Readouts", terms)
ParameterizedSystem = namedtuple("ParameterizedSystem", terms)


class Indices:
    def __init__(self, offmol: Molecule):
        self.atoms = np.array([a.molecule_atom_index for a in offmol.atoms])
        self.bonds = offmol_indices.bond_indices(offmol)
        self.angles = offmol_indices.angle_indices(offmol)
        self.propers = offmol_indices.proper_torsion_indices(offmol)
        self.impropers = offmol_indices.improper_torsion_indices(offmol)


elements = [1, 3, 6, 7, 8, 9, 15, 16, 17, 19, 35, 53]
element_encoder = OneHotEncoder(sparse=False)
element_encoder.fit(np.array(elements).reshape(-1, 1))


@lru_cache(2 ** 20)
def offmol_to_dgl(offmol: Molecule) -> dgl.DGLGraph:
    graph = dgl.from_networkx(offmol.to_networkx())
    atomic_nums = [a.element.atomic_number for a in offmol.atoms]
    X = element_encoder.transform(np.array(atomic_nums).reshape(-1, 1))
    graph.ndata["element"] = torch.Tensor(X)
    return graph


@lru_cache(2 ** 20)
def offmol_to_indices(offmol: Molecule) -> Indices:
    return Indices(offmol)


class ValenceModel(nn.Module):
    def __init__(self, node_representation: nn.Module, readouts: Readouts):
        """
        Parameters
        ----------
        node_representation is a nn.Module
            with signature
            node_representation.forward(graph, initial_node_reps) -> node_reps
        readouts contains nn.Modules as attributes
            so that readouts.angles(node_reps[:,1], node_reps[:,1])
        """
        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.readouts = readouts

    def forward(self, offmol: Molecule) -> ParameterizedSystem:
        indices = offmol_to_indices(offmol)
        graph = offmol_to_dgl(offmol)
        initial_reps = graph.ndata["element"]
        node_reps = self.node_representation.forward(graph, initial_reps)


        def symmetry_pool(f, interactions, permutations):
            permuted_reps = []
            for perm in permutations:
                rep = [node_reps[interactions[:, i]] for i in perm]
                permuted_reps.append(torch.cat(rep, dim=1))
            return sum([f(rep) for rep in permuted_reps])

        atoms = self.readouts.atoms(node_reps)

        bond_perms = [(0, 1), (1, 0)]
        bonds = symmetry_pool(self.readouts.bonds, indices.bonds, bond_perms)

        angle_perms =  [(0, 1, 2), (2, 1, 0)]
        angles = symmetry_pool(
            self.readouts.angles, indices.angles, angle_perms
        )

        # proper torsions: sum over (abcd, dcba)
        proper_perms = [(0, 1, 2, 3), (3, 2, 1, 0)]
        propers = symmetry_pool(
            self.readouts.propers, indices.propers, proper_perms
        )

        # improper torsions: sum over 3 cyclic permutations of non-central
        #   atoms, following smirnoff trefoil convention:
        #   https://github.com/openforcefield/openforcefield/blob/166c9864de3455244bd80b2c24656bd7dda3ae2d/openforcefield/typing/engines/smirnoff/parameters.py#L3326-L3360

        central = 1
        others = [0, 2, 3]
        other_perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        improper_perms = [
            (others[i], central, others[j], others[k])
            for (i, j, k) in other_perms
        ]
        if len(indices.impropers > 0):
            impropers = symmetry_pool(
                self.readouts.impropers, indices.impropers, improper_perms
            )
        else:
            impropers = torch.zeros((0, 6))

        return ParameterizedSystem(
            atoms=atoms,
            bonds=bonds,
            angles=angles,
            propers=propers,
            impropers=impropers,
        )
