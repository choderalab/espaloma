# 1. Use DGL to compute atom-representations
#   (optionally also bond and graph representations)
# 2. Symmetry-pooled readout to compute atom/bond/angle/proper/improper representations
#   (optionally also representations for coupling terms)
# 3. Functions to compute parameters from representations

import torch
from torch import nn
class ValenceModel(nn.Module):
    def __init__(self, node_representation, atom_readout, bond_readout, angle_readout,
                 proper_torsion_readout, improper_torsion_readout):
        # TODO: rather than many arguments for {atom|bond|angle|proper_torsion|improper_torsion}_readout,
        #  should these be collected in a namedtuple or other object? ReadoutModel or something?
        #  (currently manageable, but will be unmanageable when toggling coupling terms)

        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.atom_readout = atom_readout
        self.bond_readout = bond_readout
        self.angle_readout = angle_readout
        self.proper_torsion_readout = proper_torsion_readout
        self.improper_torsion_readout = improper_torsion_readout

    def forward(self, graph, inds):
        node_reps = self.node_representation.forward(graph, graph.ndata['element'])

        def symmetry_pool(f, interactions, permutations):
            return sum([f(torch.cat([node_reps[interactions[:, i]] for i in perm], dim=1)) for perm in permutations])

        atoms = self.atom_readout(node_reps)
        bonds = symmetry_pool(self.bond_readout, inds['bonds'], [(0, 1), (1, 0)])
        angles = symmetry_pool(self.angle_readout, inds['angles'], [(0, 1, 2), (2, 1, 0)])

        # proper torsions: sum over (abcd, dcba)
        proper_perms = [(0, 1, 2, 3), (3, 2, 1, 0)]
        propers = symmetry_pool(self.proper_torsion_readout, inds['propers'], proper_perms)

        # improper torsions: sum over (abcd, acdb, adbc)
        improper_perms = [(0, 1, 2, 3), (0, 3, 4, 2), (0, 4, 2, 3)]
        impropers = symmetry_pool(self.improper_torsion_readout, inds['impropers'], improper_perms)

        return dict(atoms=atoms, bonds=bonds, angles=angles, propers=propers, impropers=impropers)
