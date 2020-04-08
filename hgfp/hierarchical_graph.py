import hgfp
import dgl
import torch
from rdkit import Chem
import numpy as np


def from_oemol(mol):

    g = hgfp.graph.from_oemol(mol)

    # get the adjacency matrix of the graph
    adjacency_matrix = g.adjacency_matrix().to_dense().numpy()

    # get the number of atoms
    n_atoms = adjacency_matrix.shape[0]

    # get the bonds, angles, torsions, and pairwise indices
    (
        bond_idxs,
        angle_idxs,
        torsion_idxs,
        one_four_idxs,
        nonbonded_idxs
    ) = hgfp.mm.idxs.from_adjaceny_matrix(adjacency_matrix)

    g2_idxs = np.concatenate([bond_idxs, np.flip(bond_idxs, axis=1)], axis=0)
    g3_idxs = np.concatenate([angle_idxs, np.flip(angle_idxs, axis=1)], axis=0)
    g4_idxs = np.concatenate([torsion_idxs, np.flip(torsion_idxs, axis=1)], axis=0)

    g2_dict = {tuple(g2_idx): idx for (idx, g2_idx) in enumerate(list(g2_idxs))}
    g3_dict = {tuple(g3_idx): idx for (idx, g3_idx) in enumerate(list(g3_idxs))}

    hg = {}

    hg['g1', 'g1_in_atom', 'atom'] = np.stack(
        [
            np.arange(n_atoms),
            np.arange(n_atoms)
        ],
        axis=1)

    hg[('g2', 'g2_in_bond', 'bond')] = np.stack(
        [
            np.arange(g2_idxs.shape[0]),
            np.concatenate([np.arange(bond_idxs.shape[0]), np.arange(bond_idxs.shape[0])])
        ],
        axis=1)

    hg[('g3', 'g3_in_angle', 'angle')] = np.stack(
        [
            np.arange(g3_idxs.shape[0]),
            np.concatenate([np.arange(angle_idxs.shape[0]), np.arange(angle_idxs.shape[0])])
        ],
        axis=1)

    hg[('g4', 'g4_in_torsion', 'torsion')] = np.stack(
        [
            np.arange(g4_idxs.shape[0]),
            np.concatenate([np.arange(torsion_idxs.shape[0]), np.arange(torsion_idxs.shape[0])])
        ],
        axis=1)

    hg[('g1', 'g1_in_g', 'g')] = np.stack(
        [
            np.arange(n_atoms),
            np.zeros((n_atoms, ))
        ],
        axis=1)

    for g_idx in range(2, 5):
        hg[('g%s' % g_idx, 'g%s_in_g' % g_idx, 'g')] = np.stack(
            [
                np.arange(locals()['g%s_idxs' % g_idx].shape[0]),
                np.zeros((locals()['g%s_idxs' % g_idx].shape[0], ))
            ],
            axis=1)

    for g_idx in range(2, 5):
        for v_idx in range(g_idx):
            hg[
                (
                    'g1',
                    'g1_as_%s_in_g%s' % (v_idx, g_idx),
                    'g%s' % g_idx
                )
            ] = np.stack(
                [
                    locals()['g%s_idxs' % g_idx][:, v_idx],
                    np.arange(locals()['g%s_idxs' % g_idx].shape[0])
                ],
                axis=1)

            hg[
                (
                    'g%s' % g_idx,
                    'g%s_has_%s_g1' % (g_idx, v_idx),
                    'g1'
                )
            ] = np.stack(
                [
                    np.arange(locals()['g%s_idxs' % g_idx].shape[0]),
                    locals()['g%s_idxs' % g_idx][:, v_idx],
                ],
                axis=1)

    for g_idx in range(3, 5):
        for sub_g_idx in range(2):
            sub_g_dict = locals()['g%s_dict' % (g_idx-1)]

            hg[
                (
                    'g%s' % g_idx,
                    'g%s_has_%s_g%s' % (g_idx, sub_g_idx, g_idx - 1),
                    'g%s' % (g_idx - 1)
                )
            ] = np.stack(
                [
                    np.arange(locals()['g%s_idxs' % g_idx].shape[0]),
                    np.array([sub_g_dict[tuple(x)] for x in locals()['g%s_idxs' % g_idx][:, sub_g_idx:sub_g_idx + g_idx - 1]])
                ],
                axis=1)

            hg[
                (
                    'g%s' % (g_idx - 1),
                    'g%s_as_%s_in_g%s' % (g_idx - 1, sub_g_idx, g_idx),
                    'g%s' % (g_idx)
                )
            ] = np.stack(
                [
                    np.array([sub_g_dict[tuple(x)] for x in locals()['g%s_idxs' % g_idx][:, sub_g_idx:sub_g_idx + g_idx - 1]]),
                    np.arange(locals()['g%s_idxs' % g_idx].shape[0])
                ],
                axis=1)

    '''
    rings = mol.GetRingInfo().AtomRings()
    rings = rings + tuple([ring[::-1] for ring in rings])

    hg[('ring', 'ring_has_g1', 'g1')] = []
    hg[('g1', 'g1_in_ring', 'ring')] = []

    ring_idx = 0
    for ring in rings:
        ring_size = len(ring)
        for _ in range(ring_size - 1):
            for v in ring:
                hg[('ring', 'ring_has_g1', 'g1')].append([ring_idx, v])
                hg[('g1', 'g1_in_ring', 'ring')].append([v, ring_idx])
            ring = ring[1:] + ring[:1]
            ring_idx += 1

    '''

    elements = torch.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()])

    x =  torch.zeros(
        elements.shape[0], 100, dtype=torch.float32)

    x[
        torch.arange(x.shape[0]),
        torch.squeeze(elements).long()] = 1.0

    x = torch.cat(
        [
            x,
            torch.stack([hgfp.graph.fp_oe(atom) for atom in mol.GetAtoms()], dim=0)
        ],
        dim=1)

    hg = dgl.heterograph({k: list(v) for k, v in hg.items()})

    hg.nodes['g1'].data['h0'] = x
    hg.nodes['bond'].data['idxs'] = torch.Tensor(bond_idxs)
    hg.nodes['angle'].data['idxs'] = torch.Tensor(angle_idxs)
    hg.nodes['torsion'].data['idxs'] = torch.Tensor(torsion_idxs)

    return hg
