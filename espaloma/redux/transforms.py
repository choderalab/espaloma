import torch
torch.set_default_dtype(torch.float64)

from espaloma.redux.symmetry import ParameterizedSystem, offmol_to_indices


def null_params_from_offmol(offmol, atom_dim=2, bond_dim=2, angle_dim=2,
                            proper_dim=6, improper_dim=6):
    """generate parameters that will produce energy of 0"""

    inds = offmol_to_indices(offmol)

    atoms = torch.ones((len(inds.atoms), atom_dim))

    # initialize bonds with eq = 1, k = 0
    bonds = torch.zeros((len(inds.bonds), bond_dim))
    bonds[:, 1] = 1.0

    # initialize angles with eq = 0, k = 0
    angles = torch.zeros((len(inds.angles), angle_dim))

    # ks = 0 for propers and impropers
    propers = torch.zeros((len(inds.propers), proper_dim))
    impropers = torch.zeros((len(inds.impropers), improper_dim))

    return ParameterizedSystem(
        atoms, bonds, angles, propers, impropers
    )


def scale_params(params, scaling=1e-3):
    """TODO: if doing this nicely, should overload ParameterizedSystem.__mul__"""
    return ParameterizedSystem(
        params.atoms * scaling,
        params.bonds * scaling,
        params.angles * scaling,
        params.propers * scaling,
        params.impropers * scaling
    )


def shift_params(params, ref_params):
    """TODO: if doing this nicely, should overload ParameterizedSystem.__add__"""
    return ParameterizedSystem(
        params.atoms + ref_params.atoms,
        params.bonds + ref_params.bonds,
        params.angles + ref_params.angles,
        params.propers + ref_params.propers,
        params.impropers + ref_params.impropers
    )


def perturb_from_null_params(offmol, raw_params, scaling=1e-3):
    scaled_params = scale_params(raw_params, scaling)

    null_params = null_params_from_offmol(offmol)

    return shift_params(scaled_params, null_params)
