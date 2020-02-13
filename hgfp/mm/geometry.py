# =============================================================================
# imports
# =============================================================================
import torch


# =============================================================================
# utility functions
# =============================================================================
def get_distance_matrix(coordinates):
    """ Calculate the distance matrix from coordinates.

    $$
    D_{ij}^2 = <X_i, X_i> - 2<X_i, X_j> + <X_j, X_j>
    $$
    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)

    """
    X_2 = torch.sum(
        torch.pow(
            coordinates,
            2),
        axis=2,
        keepdim=True)

    return torch.sqrt(
        torch.nn.functional.relu(
            X_2 - 2 * torch.matmul(
                coordinates,
                torch.transpose(coordinates, 2, 1)) \
                + torch.transpose(X_2, 2, 1)))


# =============================================================================
# module functions
# =============================================================================
def get_distances(idxs, coordinates):
    """ Get the distances between nodes given coordinates.

    """
    # get the distance matrix
    distance_matrix = get_distance_matrix(
        coordinates)

    # put the start and end of bonds in two arrays
    idxs_left = idxs[:, 0]
    idxs_right = idxs[:, 1]

    distances = distance_matrix[:, idxs_left, idxs_right]

    return distances


def get_angles(angle_idxs, coordinates, return_cos=False):
    """ Calculate angles from a set of indices and coordinates.

    """
    # get the coordinates of the atoms forming the angle
    # (batch_size, n_angles, 3, 3)
    angle_coordinates = coordinates[:, angle_idxs]

    # (batcn_angles, 3)
    angle_left = angle_coordinates[:, :, 1, :] \
        - angle_coordinates[:, :, 0, :]

    # (n_angles, 3)
    angle_right = angle_coordinates[:, :, 1, :] \
        - angle_coordinates[:, :, 2, :]

    # (n_batch, n_angles, )
    angles = torch.atan2(
        torch.norm(
            torch.cross(
                angle_left,
                angle_right),
            dim=2),
        torch.sum(
            torch.mul(
                angle_left,
                angle_right),
            dim=2))

    return angles

def get_torsions(torsion_idxs, coordinates, return_cos=False):
    """ Calculate the dihedrals based on coordinates and the indices of
    the torsions.

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)
    torsion_idxs: # TODO: describe
    """
    # get the coordinates of the atoms forming the dihedral
    # (batch_size, n_torsions, 4, 3)
    torsion_idxs = coordinates[:, torsion_idxs]

    # (batch_size, n_torsions, 3)
    normal_left = torch.cross(
        torsion_idxs[:, :, 1, :] - torsion_idxs[:, :, 0, :],
        torsion_idxs[:, :, 1, :] - torsion_idxs[:, :, 2, :])

    # (batch_size, n_torsions, 3)
    normal_right = torch.cross(
        torsion_idxs[:, :, 2, :] - torsion_idxs[:, :, 3, :],
        torsion_idxs[:, :, 2, :] - torsion_idxs[:, :, 1, :])

    # (batch_size, n_torsions, )
    dihedrals = torch.atan2(
        torch.norm(
            tf.linalg.cross(
                normal_left,
                normal_right),
            dim=2),
        torch.sum(
            torch.mul(
                normal_left,
                normal_right),
            dim=2))

    return dihedrals
