# =============================================================================
# imports
# =============================================================================
import numpy as np

# =============================================================================
# module functions
# =============================================================================
def get_idxs(adjacency_map):

    # bond idxs are where adjacency matrix is greater than zero.
    bond_idxs =  np.stack(
        np.where(
            np.greater(
                np.triu(
                    adjacency_map),
                0.)),
        axis=1)

    # get the bond idxs in both order
    _bond_idxs = np.concatenate(
        [
            bond_idxs,
            np.flip(
                bond_idxs,
                axis=[1])
        ],
        axis=0)

    # get the number of _directed_ bonds
    _n_bonds = np.shape(_bond_idxs)[0]

    # enumerate all bond pairs
    # (n_bond * n_bond, 4)
    bond_pairs = np.reshape(
        np.concatenate(
            [
                np.tile(
                    np.expand_dims(
                        _bond_idxs,
                        axis=0),
                    [_n_bonds, 1, 1]),
                np.tile(
                    np.expand_dims(
                        _bond_idxs,
                        axis=1),
                    [1, _n_bonds, 1])
            ],
            axis=2),
        [-1, 4])

    # angles are where two _directed bonds share one _inner_ atom
    angle_idxs = np.take(
            bond_pairs[
            np.logical_and(
                np.equal(
                    bond_pairs[:, 1],
                    bond_pairs[:, 2]),
                    np.less(
                        bond_pairs[:, 0],
                        bond_pairs[:, 3]))],
        [0, 1, 3],
        axis=1)

    # get the angle pairs in both order
    _angle_idxs = np.concatenate(
        [
            angle_idxs,
            np.flip(
                angle_idxs,
                axis=[1])
        ],
        axis=0)

    # get the number of _directed_ angles
    _n_angles = np.shape(_angle_idxs)[0]

    # enumerate all bond pairs
    # (n_angles * n_angles, 6)
    angle_pairs = np.reshape(
        np.concatenate(
            [
                np.tile(
                    np.expand_dims(
                        _angle_idxs,
                        axis=0),
                    [_n_angles, 1, 1]),
                np.tile(
                    np.expand_dims(
                        _angle_idxs,
                        axis=1),
                    [1, _n_angles, 1])
            ],
            axis=2),
        [-1, 6])

    # angles are where two _directed bonds share one _inner_ atom
    torsion_idxs = np.take(
            angle_pairs[
            np.logical_and(
                np.logical_and(
                    np.equal(
                        angle_pairs[:, 1],
                        angle_pairs[:, 3]),
                    np.equal(
                        angle_pairs[:, 2],
                        angle_pairs[:, 4])),
                np.less(
                    angle_pairs[:, 0],
                    angle_pairs[:, 5]))],
        [0, 1, 2, 5],
        axis=1)

    # one four idxs are just the two ends of torsion idxs
    one_four_idxs = np.take(
        torsion_idxs,
        [0, 3],
        axis=1)

    # nonbonded idxs are those that cannot be connected by
    # 1-, 2-, and 3-walks
    adjacency_map_full = np.add(
        adjacency_map,
        np.transpose(
            adjacency_map))

    nonbonded_idxs = np.stack(
        np.where(
            np.equal(
                np.sum(
                    [
                        # 1-walk
                        adjacency_map_full,

                        # 2-walk
                        np.matmul(
                            adjacency_map_full,
                            adjacency_map_full),

                        # 3-walk
                        np.matmul(
                            adjacency_map_full,
                            np.matmul(
                                adjacency_map_full,
                                adjacency_map_full))
                    ],
                    axis=0),
                0.)),
        axis=1)

    return bond_idxs, angle_idxs, torsion_idxs, one_four_idxs, nonbonded_idxs
