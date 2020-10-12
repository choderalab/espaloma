# 1. Use DGL to compute atom-representations
#   (optionally also bond and graph representations)
# 2. Symmetry-encoded readout to compute atom/bond/angle/proper/improper representations
#   (optionally also representations for coupling terms)
# 3. Functions to compute parameters from representations

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import TAGConv
from torch import nn


class ValenceModel(nn.Module):
    def __init__(self, node_representation, atom_readout, bond_readout, angle_readout, proper_torsion_readout,
                 improper_torsion_readout):
        # TODO: rather than many arguments for {atom|bond|angle|proper_torsion|improper_torsion}_readout,
        #  should these be collected in a dictionary or other object? ReadoutModel or something?
        #  (currently manageable, but will be unmanageable when toggling coupling terms)

        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.atom_readout = atom_readout
        self.bond_readout = bond_readout
        self.angle_readout = angle_readout
        self.proper_torsion_readout = proper_torsion_readout
        self.improper_torsion_readout = improper_torsion_readout

    def forward(self, graph, inds):
        bond_inds, angle_inds, proper_inds, improper_inds = inds['bonds'], inds['angles'], inds['propers'], inds[
            'impropers']
        node_reps = self.node_representation.forward(graph, graph.ndata['element'])

        # TODO: write a little function that sums over specified permutations in a less noisy way

        # atoms
        atoms = self.atom_readout(node_reps)

        # bonds
        a, b = node_reps[bond_inds[:, 0]], node_reps[bond_inds[:, 1]]
        ab, ba = torch.cat((a, b), dim=1) + torch.cat((b, a), dim=1)
        bonds = self.bond_readout(ab) + self.bond_readout(ba)

        # angles
        a, b, c = node_reps[angle_inds[:, 0]], node_reps[angle_inds[:, 1]], node_reps[angle_inds[:, 2]]
        abc, cba = torch.cat((a, b, c), dim=1), torch.cat((c, b, a), dim=1)
        angles = self.angle_readout(abc) + self.angle_readout(cba)

        # proper torsions: sum over (abcd, dcba)
        a, b, c, d = node_reps[proper_inds[:, 0]], node_reps[proper_inds[:, 1]], \
                     node_reps[proper_inds[:, 2]], node_reps[proper_inds[:, 3]]
        abcd, dcba = torch.cat((a, b, c, d), dim=1), torch.cat((d, c, b, a), dim=1)
        propers = self.proper_torsion_readout(abcd) + self.proper_torsion_readout(abcd)

        # improper torsions: sum over (abcd, acdb, adbc)
        a, b, c, d = node_reps[improper_inds[:, 0]], node_reps[improper_inds[:, 1]], \
                     node_reps[improper_inds[:, 2]], node_reps[improper_inds[:, 3]]
        abcd, acdb, adbc = torch.cat((a, b, c, d), dim=1), torch.cat((a, c, d, b), dim=1), torch.cat((a, d, b, c),
                                                                                                     dim=1)
        impropers = self.torsion_readout(abcd) + self.torsion_readout(acdb) + self.torsion_readout(adbc)

        return dict(atoms=atoms, bonds=bonds, angles=angles, propers=propers, impropers=impropers)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class TAG(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, k=2, activation=F.relu):
        super(TAG, self).__init__()
        self.layer1 = TAGConv(in_feats, h_feats, k, activation=activation)
        self.layer2 = TAGConv(h_feats, h_feats, k, activation=activation)
        self.layer3 = TAGConv(h_feats, num_classes, k, activation=activation)
        self.activation = activation

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = self.activation(h)
        h = self.layer2(graph, h)
        h = self.activation(h)
        h = self.layer3(graph, h)
        return h
