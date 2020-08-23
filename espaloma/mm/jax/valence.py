from jax.config import config
config.update("jax_enable_x64", True)

from jax import numpy as np
import numpy as onp


def k_to_omm_unit(k):
    """convert a spring constant from hartree/mol / angstrom^2 to kJ/mol / nm^2"""
    return 262549.9639479825 * k



# Can represent springs a couple different ways
# 1. In terms of spring constant and equilibrium length
#   u(r; k, r0) = k/2 (r - r0)^2
# 2. In terms of a fixed interval of allowable equilibrium lengths [r1, r2]
#   u(r; k1, k2) = k1 (r - r1)^2 + k2 (r - r2)^2
#   This has the benefit of allowing us to express variable equilibrium length in terms of
#   fixed design matrix [(r - r1)^2, (r - r2)^2] and variable coefficients [k1, k2].
from jax import jit


@jit
def harmonic_bond_potential_alt_param(r, k1, k2, r1, r2):
    return k1 * (r - r1) ** 2 + k2 * (r - r2) ** 2


r1, r2 = 0.0, 0.2


@jit
def harmonic_angle_potential_alt_param(theta, k1, k2, theta1, theta2):
    return k1 * (theta - theta1) ** 2 + k2 * (theta - theta2) ** 2


def compute_harmonic_bond_potential(xyz, params, pair_inds, bond_inds):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param pair_inds:
        array of shape (len(offmol.bonds), 2)
    :param bond_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    ks, r0s = params[:n_unique], params[n_unique:]
    k, r0 = ks[bond_inds], r0s[bond_inds]

    r = compute_distances(xyz, pair_inds)
    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


def compute_harmonic_bond_potential_alt_param(xyz, params, pair_inds, bond_inds):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param pair_inds:
        array of shape (len(offmol.bonds), 2)
    :param bond_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    k1s, k2s = params[:n_unique], params[n_unique:]
    k1, k2 = k1s[bond_inds], k2s[bond_inds]

    k = k1 + k2
    r0 = ((k1 * r1) + (k2 * r2)) / (k1 + k2)

    r = compute_distances(xyz, pair_inds)


    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


def compute_harmonic_angle_potential(xyz, params, triple_inds, angle_inds):
    """

    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param triple_inds:
        array of shape (len(offmol.angles), 3)
    :param angle_inds:
        numpy array of length len(offmol.angles),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    ks, theta0s = params[:n_unique], params[n_unique:]
    # Jax: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    #angle_inds = tuple(angle_inds)
    angle_inds = onp.array(angle_inds)
    #print(ks)
    #print(angle_inds)

    k = ks[angle_inds]
    theta0 = theta0s[angle_inds]
    # theta0s: size 23
    # params: size 46
    # ks: size 23

    theta = compute_angles(xyz, triple_inds)
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)


theta1, theta2 = 0, 2 * np.pi


def compute_harmonic_angle_potential_alt_param(xyz, params, triple_inds, angle_inds):
    """
    :param xyz:
    :param params:
        array of length 2 * n_unique
    :param triple_inds:
        array of shape (len(offmol.bonds), 2)
    :param angle_inds:
        numpy array of length len(offmol.bonds),
        taking integer values in range 0 through n_unique
    :return:
    """

    n_unique = int(len(params) / 2)
    k1s, k2s = params[:n_unique], params[n_unique:]
    k1, k2 = k1s[angle_inds], k2s[angle_inds]

    k = k1 + k2
    theta0 = ((k1 * theta1) + (k2 * theta2)) / (k1 + k2)

    theta = compute_angles(xyz, triple_inds)
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)


n_periodicities = 6
periodicities = np.arange(n_periodicities) + 1


def compute_periodic_torsion_potential(xyz, params, quad_inds, torsion_inds):
    """

    :param xyz:
    :param params:
        length ( 2 * n_unique * n_periodicities )
    :param quad_inds:
    :param torsion_inds:
    :return:
    """
    theta = compute_torsions(xyz, quad_inds)

    n_unique = int(len(params) / (2 * n_periodicities))
    params = np.reshape(params, (n_unique, (2 * n_periodicities)))

    ks, phases = params[torsion_inds][:, :n_periodicities], params[torsion_inds][:, n_periodicities:]

    # TODO; clean this up a bit
    periodicities_ = np.array([periodicities for _ in ks])

    return np.sum(periodic_torsion_potential(theta, ks, phases, periodicities_), axis=1)




from functools import lru_cache
@lru_cache(maxsize=len(offmols))
def extract_bond_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    b12 = bond_features_from_offmol(offmol)

    a1_inds = onp.array([bond.atom1_index for bond in offmol.bonds])
    a2_inds = onp.array([bond.atom2_index for bond in offmol.bonds])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]

    x = np.hstack((a1, b12, a2))
    inds = np.vstack((a1_inds, a2_inds)).T

    return x, inds


@lru_cache(maxsize=len(offmols))
def extract_angle_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    bond_features = bond_features_from_offmol(offmol)
    bond_index_dict = bond_index_dict_from_offmol(offmol)

    angles = list(offmol.angles)  # offmol.angles is a set of 3-tuples of atoms

    a1_inds = onp.array([a.molecule_atom_index for (a, _, _) in angles])
    a2_inds = onp.array([b.molecule_atom_index for (_, b, _) in angles])
    a3_inds = onp.array([c.molecule_atom_index for (_, _, c) in angles])

    b12_inds = onp.array([bond_index_dict[(a.molecule_atom_index, b.molecule_atom_index)] for (a, b, _) in angles])
    b23_inds = onp.array([bond_index_dict[(b.molecule_atom_index, c.molecule_atom_index)] for (_, b, c) in angles])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]
    a3 = atoms[a3_inds]

    b12 = bond_features[b12_inds]
    b23 = bond_features[b23_inds]

    x = np.hstack((a1, b12, a2, b23, a3))
    inds = np.vstack((a1_inds, a2_inds, a3_inds)).T

    return x, inds


@lru_cache(maxsize=len(offmols))
def extract_torsion_term_inputs(offmol):
    atoms = atom_features_from_offmol(offmol)

    bond_features = bond_features_from_offmol(offmol)
    bond_index_dict = bond_index_dict_from_offmol(offmol)

    torsions = list(offmol.propers)  # offmol.propers is a set of 4-tuples of atoms

    a1_inds = onp.array([a.molecule_atom_index for (a, _, _, _) in torsions])
    a2_inds = onp.array([b.molecule_atom_index for (_, b, _, _) in torsions])
    a3_inds = onp.array([c.molecule_atom_index for (_, _, c, _) in torsions])
    a4_inds = onp.array([d.molecule_atom_index for (_, _, _, d) in torsions])

    b12_inds = onp.array([bond_index_dict[(a.molecule_atom_index, b.molecule_atom_index)] for (a, b, _, _) in torsions])
    b23_inds = onp.array([bond_index_dict[(b.molecule_atom_index, c.molecule_atom_index)] for (_, b, c, _) in torsions])
    b34_inds = onp.array([bond_index_dict[(c.molecule_atom_index, d.molecule_atom_index)] for (_, _, c, d) in torsions])

    a1 = atoms[a1_inds]
    a2 = atoms[a2_inds]
    a3 = atoms[a3_inds]
    a4 = atoms[a3_inds]

    b12 = bond_features[b12_inds]
    b23 = bond_features[b23_inds]
    b34 = bond_features[b34_inds]

    x = np.hstack((a1, b12, a2, b23, a3, b34, a4))
    inds = np.vstack((a1_inds, a2_inds, a3_inds, a4_inds)).T

    return x, inds

def compute_harmonic_bond_potential(offmol, xyz, f_2_params):
    x, inds = extract_bond_term_inputs(offmol)
    r = compute_distances(xyz, inds)
    k_, r0 = f_bond(f_2_params, x).T
    k = k_to_omm_unit(k_)
    #k = np.exp(log_k)
    return np.sum(harmonic_bond_potential(r, k, r0), axis=1)


def compute_harmonic_angle_potential(offmol, xyz, f_3_params):
    x, inds = extract_angle_term_inputs(offmol)
    theta = compute_angles(xyz, inds)
    k_, theta0 = f_angle(f_3_params, x).T
    #k = np.exp(log_k)
    k = k_to_omm_unit(k_)
    return np.sum(harmonic_angle_potential(theta, k, theta0), axis=1)


def compute_periodic_torsion_potential(offmol, xyz, f_4_params):
    x, inds = extract_torsion_term_inputs(offmol)
    theta = compute_torsions(xyz, inds)

    params = f_torsion(f_4_params, x)
    #ks, phases = params[:, :n_periodicities], params[:, n_periodicities:]
    #log_ks = params
    ks_ = params
    #ks = np.exp(log_ks)
    ks = k_to_omm_unit(ks_)
    phases = np.zeros_like(ks)

    # TODO; clean this up a bit
    periodicities_ = np.array([periodicities for _ in ks])

    return np.sum(periodic_torsion_potential(theta, ks, phases, periodicities_), axis=1)

# Altogether
def pred_valence_energy(offmol, xyz, f_2_params, f_3_params, f_4_params):
    bond_energy = compute_harmonic_bond_potential(offmol, xyz, f_2_params)
    angle_energy = compute_harmonic_angle_potential(offmol, xyz, f_3_params)
    torsion_energy = compute_periodic_torsion_potential(offmol, xyz, f_4_params)

    return bond_energy + angle_energy + torsion_energy


def pred_valence_force(offmol, xyz, f_2_params, f_3_params, f_4_params):
    return grad(lambda xyz : np.sum(pred_valence_energy(offmol, xyz, f_2_params, f_3_params, f_4_params)))(xyz)
