# =============================================================================
# imports
# =============================================================================
import torch
import dgl
import hgfp


# =============================================================================
# module functions
# =============================================================================
def from_graph(g):
    """ Function to convert molecular graph to heterograph, with nodes
    representing bonds, angles, torsions, and pairwise interactions.

    """
    # initialize heterograph
    hg = {
        ('bond', 'bond_has_atom', 'atom'):[],
        ('atom', 'atom_neighbors_atom', 'atom'): [],
        ('angle', 'angle_has_center_atom', 'atom'): [],
        ('angle', 'angle_has_side_atom', 'atom'): [],
        ('atom', 'atom_in_bond', 'bond'): [],

        ('atom', 'atom_as_center_in_angle', 'angle'): [],
        ('atom', 'atom_as_side_in_angle', 'angle'): [],

        ('torsion', 'torsion_has_0_atom', 'atom'): [],
        ('torsion', 'torsion_has_1_atom', 'atom'): [],
        ('torsion', 'torsion_has_2_atom', 'atom'): [],
        ('torsion', 'torsion_has_3_atom', 'atom'): [],
        ('atom', 'atom_as_0_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_1_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_2_in_torsion', 'torsion'): [],
        ('atom', 'atom_as_3_in_torsion', 'torsion'): [],


        ('atom', 'atom_in_one_four', 'one_four'): [],
        ('one_four', 'one_four_has_atom', 'atom'): [],

        ('atom', 'atom_in_nonbonded', 'nonbonded'): [],
        ('nonbonded', 'nonbonded_has_atom', 'atom'): []}

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

    # add bonds
    for idx in range(bond_idxs.shape[0]):
        hg[('bond', 'bond_has_atom', 'atom')].append((idx, bond_idxs[idx, 0]))
        hg[('bond', 'bond_has_atom', 'atom')].append((idx, bond_idxs[idx, 1]))
        hg[('atom', 'atom_neighbors_atom', 'atom')].append((bond_idxs[idx, 0], bond_idxs[idx, 1]))
        hg[('atom', 'atom_neighbors_atom', 'atom')].append((bond_idxs[idx, 1], bond_idxs[idx, 0]))
        hg[('atom', 'atom_in_bond', 'bond')].append((bond_idxs[idx, 0], idx))
        hg[('atom', 'atom_in_bond', 'bond')].append((bond_idxs[idx, 1], idx))

    # add angles
    for idx in range(angle_idxs.shape[0]):
        hg[('angle', 'angle_has_center_atom', 'atom')].append((idx, angle_idxs[idx, 1]))
        hg[('angle', 'angle_has_side_atom', 'atom')].append((idx, angle_idxs[idx, 0]))
        hg[('angle', 'angle_has_side_atom', 'atom')].append((idx, angle_idxs[idx, 2]))
        hg[('atom', 'atom_as_center_in_angle', 'angle')].append((angle_idxs[idx, 1], idx))
        hg[('atom', 'atom_as_side_in_angle', 'angle')].append((angle_idxs[idx, 0], idx))
        hg[('atom', 'atom_as_side_in_angle', 'angle')].append((angle_idxs[idx, 2], idx))

    # add torsions
    for idx in range(torsion_idxs.shape[0]):
        hg[('torsion', 'torsion_has_0_atom', 'atom')].append((idx, torsion_idxs[idx, 0]))
        hg[('torsion', 'torsion_has_1_atom', 'atom')].append((idx, torsion_idxs[idx, 1]))
        hg[('torsion', 'torsion_has_2_atom', 'atom')].append((idx, torsion_idxs[idx, 2]))
        hg[('torsion', 'torsion_has_3_atom', 'atom')].append((idx, torsion_idxs[idx, 3]))
        hg[('atom', 'atom_as_0_in_torsion', 'torsion')].append((torsion_idxs[idx, 0], idx))
        hg[('atom', 'atom_as_1_in_torsion', 'torsion')].append((torsion_idxs[idx, 1], idx))
        hg[('atom', 'atom_as_2_in_torsion', 'torsion')].append((torsion_idxs[idx, 2], idx))
        hg[('atom', 'atom_as_3_in_torsion', 'torsion')].append((torsion_idxs[idx, 3], idx))

    # add one_four
    for idx in range(one_four_idxs.shape[0]):
        hg[('one_four', 'one_four_has_atom', 'atom')].append((idx, one_four_idxs[idx, 0]))
        hg[('one_four', 'one_four_has_atom', 'atom')].append((idx, one_four_idxs[idx, 1]))
        hg[('atom', 'atom_in_one_four', 'one_four')].append((one_four_idxs[idx, 0], idx))
        hg[('atom', 'atom_in_one_four', 'one_four')].append((one_four_idxs[idx, 1], idx))

    for idx in range(nonbonded_idxs.shape[0]):
        hg[('nonbonded', 'nonbonded_has_atom', 'atom')].append((idx, nonbonded_idxs[idx, 0]))
        hg[('nonbonded', 'nonbonded_has_atom', 'atom')].append((idx, nonbonded_idxs[idx, 1]))
        hg[('atom', 'atom_in_nonbonded', 'nonbonded')].append((nonbonded_idxs[idx, 0], idx))
        hg[('atom', 'atom_in_nonbonded', 'nonbonded')].append((nonbonded_idxs[idx, 1], idx))

    hg = dgl.heterograph(hg)

    # put all atom data into heterograph
    hg.nodes['atom'].data['type'] = g.ndata['type']

    try:
        hg.nodes['atom'].data['xyz'] = g.ndata['xyz']
    except:
        pass

    hg.edges['atom_neighbors_atom'].data['type'] = g.edata['type']

    return hg
