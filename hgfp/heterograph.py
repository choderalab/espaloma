# =============================================================================
# imports
# =============================================================================
import torch
import dgl
import hgfp
import numpy as np


# =============================================================================
# module functions
# =============================================================================
def from_graph(g):
    """ Function to convert molecular graph to heterograph, with nodes
    representing bonds, angles, torsions, and pairwise interactions.

    """
    # initialize heterograph
    hg = {
        # ('bond', 'bond_has_atom', 'atom'):[],
        ('atom', 'atom_neighbors_atom', 'atom'): [],
        # ('angle', 'angle_has_center_atom', 'atom'): [],
        # ('angle', 'angle_has_side_atom', 'atom'): [],
        ('atom', 'atom_in_bond', 'bond'): [],

        ('atom', 'atom_as_center_in_angle', 'angle'): [],
        ('atom', 'atom_as_side_in_angle', 'angle'): [],

        # ('torsion', 'torsion_has_0_atom', 'atom'): [],
        # ('torsion', 'torsion_has_1_atom', 'atom'): [],
        # ('torsion', 'torsion_has_2_atom', 'atom'): [],
        # ('torsion', 'torsion_has_3_atom', 'atom'): [],
        ('atom', 'atom_as_0_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_1_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_2_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_3_in_torsion', 'torsion'): [],


        ('atom', 'atom_in_one_four', 'one_four'): [],
        # ('one_four', 'one_four_has_atom', 'atom'): [],

        ('atom', 'atom_in_nonbonded', 'nonbonded'): [],
        # ('nonbonded', 'nonbonded_has_atom', 'atom'): [],

        ('atom', 'atom_in_mol', 'mol'): [],
        ('bond', 'bond_in_mol', 'mol'): [],
        ('angle', 'angle_in_mol', 'mol'): [],
        ('torsion', 'torsion_in_mol', 'mol'): [],
        ('one_four', 'one_four_in_mol', 'mol'): [],
        ('nonbonded', 'nonbonded_in_mol', 'mol'): []

        }

    # get the adjacency matrix of the graph
    adjacency_matrix = g.adjacency_matrix().to_dense().numpy()

    # get the bonds, angles, torsions, and pairwise indices
    (
        bond_idxs,
        angle_idxs,
        torsion_idxs,
        one_four_idxs,
        nonbonded_idxs
    ) = hgfp.mm.idxs.from_adjaceny_matrix(adjacency_matrix)

    # add self loop
    hg[('atom', 'atom_is_self_atom', 'atom')] = np.stack(
        [
            np.arange(adjacency_matrix.shape[0]),
            np.arange(adjacency_matrix.shape[0])
        ],
        axis=1)

    # atom neighboring
    # for message passing
    hg[('atom', 'atom_neighbors_atom', 'atom')] = np.argwhere(
        np.greater(adjacency_matrix, 0))

    # add bonds
    hg[('atom', 'atom_as_0_in_bond', 'bond')] = np.stack(
        [
            bond_idxs[:, 0],
            np.arange(bond_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_1_in_bond', 'bond')] = np.stack(
        [
            bond_idxs[:, 1],
            np.arange(bond_idxs.shape[0])
        ],
        axis=1)

    hg[('bond', 'bond_has_atom', 'atom')] = np.concatenate(
        [
            np.stack(
                [
                    np.arange(bond_idxs.shape[0]),
                    bond_idxs[:, 0]
                ],
                axis=1),
            np.stack(
                [
                    np.arange(bond_idxs.shape[0]),
                    bond_idxs[:, 1]
                ],
                axis=1),
        ],
        axis=0)

    # add angles
    # hg[('atom', 'atom_as_center_in_angle', 'angle')] = np.stack(
    #     [
    #         angle_idxs[:, 1],
    #         np.arange(angle_idxs.shape[0])
    #     ],
    #     axis=1)
    #
    # hg[('atom', 'atom_as_side_in_angle', 'angle')] = np.concatenate(
    #     [
    #         np.stack(
    #             [
    #                 angle_idxs[:, 0],
    #                 np.arange(angle_idxs.shape[0])
    #             ],
    #             axis=1),
    #         np.stack(
    #             [
    #                 angle_idxs[:, 2],
    #                 np.arange(angle_idxs.shape[0])
    #             ],
    #             axis=1)
    #     ],
    #     axis=0)

    # add angles
    hg[('atom', 'atom_as_0_in_angle', 'angle')] = np.stack(
        [
            angle_idxs[:, 0],
            np.arange(angle_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_1_in_angle', 'angle')] = np.stack(
        [
            angle_idxs[:, 1],
            np.arange(angle_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_2_in_angle', 'angle')] = np.stack(
        [
            angle_idxs[:, 2],
            np.arange(angle_idxs.shape[0])
        ],
        axis=1)


    # add torsions
    hg[('atom', 'atom_as_0_in_torsion', 'torsion')] = np.stack(
        [
            torsion_idxs[:, 0],
            np.arange(torsion_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_1_in_torsion', 'torsion')] = np.stack(
        [
            torsion_idxs[:, 1],
            np.arange(torsion_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_2_in_torsion', 'torsion')] = np.stack(
        [
            torsion_idxs[:, 2],
            np.arange(torsion_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_as_3_in_torsion', 'torsion')] = np.stack(
        [
            torsion_idxs[:, 3],
            np.arange(torsion_idxs.shape[0])
        ],
        axis=1)

    hg[('atom', 'atom_in_one_four', 'one_four')] = np.concatenate(
        [
            np.stack(
                [
                    one_four_idxs[:, 0],
                    np.arange(one_four_idxs.shape[0])
                ],
                axis=1),
            np.stack(
                [
                    one_four_idxs[:, 1],
                    np.arange(one_four_idxs.shape[0])
                ],
                axis=1)
        ],
        axis=0)

    hg[('atom', 'atom_in_nonbonded', 'nonbonded')] = np.concatenate(
        [
            np.stack(
                [
                    nonbonded_idxs[:, 0],
                    np.arange(nonbonded_idxs.shape[0])
                ],
                axis=1),
            np.stack(
                [
                    nonbonded_idxs[:, 1],
                    np.arange(nonbonded_idxs.shape[0])
                ],
                axis=1),
        ],
        axis=0)

    # add bonds
    hg[('atom', 'atom_in_mol', 'mol')] = np.stack(
        [
            np.arange(adjacency_matrix.shape[0]),
            np.zeros((adjacency_matrix.shape[0], ))
        ],
        axis=1)

    hg[('mol', 'mol_has_atom', 'atom')] = np.stack(
        [
            np.zeros((adjacency_matrix.shape[0], )),
            np.arange(adjacency_matrix.shape[0])
        ],
        axis=1)

    hg[('bond', 'bond_in_mol', 'mol')] = np.stack(
        [
            np.arange(bond_idxs.shape[0]),
            np.zeros((bond_idxs.shape[0], ))
        ],
        axis=1)

    hg[('angle', 'angle_in_mol', 'mol')] = np.stack(
        [
            np.arange(angle_idxs.shape[0]),
            np.zeros((angle_idxs.shape[0], ))
        ],
        axis=1)

    hg[('torsion', 'torsion_in_mol', 'mol')] = np.stack(
        [
            np.arange(torsion_idxs.shape[0]),
            np.zeros((torsion_idxs.shape[0], ))
        ],
        axis=1)

    hg[('one_four', 'one_four_in_mol', 'mol')] = np.stack(
        [
            np.arange(one_four_idxs.shape[0]),
            np.zeros((one_four_idxs.shape[0], ))
        ],
        axis=1)

    hg[('nonbonded', 'nonbonded_in_mol', 'mol')] = np.stack(
        [
            np.arange(nonbonded_idxs.shape[0]),
            np.zeros((nonbonded_idxs.shape[0], ))
        ],
        axis=1)

    bond_idxs_dict = {tuple(idxs): idx for idx, idxs in enumerate(list(bond_idxs))}
    
    bond_idxs_dict.update(
         {tuple(np.flip(idxs)): idx for idx, idxs in enumerate(list(bond_idxs))})
    
    angle_idxs_dict = {tuple(idxs): idx for idx, idxs in enumerate(list(angle_idxs))}
    
    angle_idxs_dict.update(
        {tuple(np.flip(idxs)): idx for idx, idxs in enumerate(list(angle_idxs))})

    for term_idx in range(2):

        hg[('bond', 'bond_as_%s_in_angle' % term_idx, 'angle')] = np.stack(
            [
                np.array([bond_idxs_dict[tuple(idxs)] for idxs in angle_idxs[:, term_idx:(term_idx+2)]]),
                np.arange(angle_idxs.shape[0])
            ],
            axis=1)


    for term_idx in range(3):

        hg[('bond', 'bond_as_%s_in_torsion' % term_idx, 'torsion')] = np.stack(
            [
                np.array([bond_idxs_dict[tuple(idxs)] for idxs in torsion_idxs[:, term_idx:(term_idx+2)]]),
                np.arange(torsion_idxs.shape[0])
            ],
            axis=1)


    for term_idx in range(2):

        hg[('angle', 'angle_as_%s_in_torsion' % term_idx, 'torsion')] = np.stack(
            [
                np.array([angle_idxs_dict[tuple(idxs)] for idxs in torsion_idxs[:, term_idx:(term_idx+3)]]),
                np.arange(torsion_idxs.shape[0])
            ],
            axis=1)

    hg = dgl.heterograph({k: list(v) for k, v in hg.items()})

    # put all atom data into heterograph
    hg.nodes['atom'].data['type'] = g.ndata['type']
    hg.nodes['atom'].data['h0'] = g.ndata['h0']


    # put indices in bonds, angles, and torsions
    hg.nodes['bond'].data['idxs'] = torch.Tensor(bond_idxs)
    hg.nodes['angle'].data['idxs'] = torch.Tensor(angle_idxs)
    hg.nodes['torsion'].data['idxs'] = torch.Tensor(torsion_idxs)

    try:
        hg.nodes['atom'].data['xyz'] = g.ndata['xyz']
    except:
        pass

    hg.edges['atom_neighbors_atom'].data['type'] = g.edata['type']

    hg.nodes['bond'].data['type'] = torch.chunk(
        g.edata['type'],
        2,
        dim=0)[0]

    return hg
