from simtk import unit
import numpy as np
import torch

# lennard jones
eps_unit = unit.kilocalorie_per_mole
rmin_half_unit = unit.angstrom

def extract_lj(labeled_mol):
    vdw = labeled_mol['vdW']
    atom_inds = list(vdw.keys())
    eps = np.array([vdw[ind].epsilon / eps_unit for ind in atom_inds])
    rmin_half = np.array([vdw[ind].rmin_half / rmin_half_unit for ind in atom_inds])
    return eps, rmin_half

# bonds
length_unit = unit.angstrom
bond_opt_k_unit = unit.kilocalorie_per_mole / (length_unit**2) * 1000
bond_report_k_unit = unit.kilocalorie_per_mole / (length_unit**2)
# different units for optimization and reporting

def extract_bond_params(labeled_mol):
    bonds = labeled_mol['Bonds']
    bond_inds = list(bonds.keys())
    length = np.array([bonds[ind].length / length_unit for ind in bond_inds])
    bond_k = np.array([bonds[ind].k / bond_opt_k_unit for ind in bond_inds])
    return np.array(bond_inds), length, bond_k

# angles
angle_unit = unit.radian
angle_k_report_unit = (unit.kilocalorie_per_mole / unit.radian**2)
angle_k_opt_unit = angle_k_report_unit * 250

def extract_angle_params(labeled_mol):
    angles = labeled_mol['Angles']
    angle_inds = list(angles.keys())
    eq_angles = np.array([angles[ind].angle / angle_unit for ind in angle_inds])
    angle_k = np.array([angles[ind].k / angle_k_opt_unit for ind in angle_inds])
    return np.array(angle_inds), eq_angles, angle_k

# torsions
energy_unit = unit.kilocalorie_per_mole
def extract_torsion_params(labeled_mol):
    torsions = labeled_mol['ProperTorsions']
    torsion_inds = list(torsions.keys())
    N_torsions = len(torsion_inds)
    
    torsion_ks = np.zeros((N_torsions, 6))
    torsion_phases = np.zeros((N_torsions, 6))
    
    for i, ind in enumerate(torsion_inds):
        torsion = torsions[ind]
        for j, p in enumerate(torsion.periodicity):
            torsion_ks[i, p - 1] = torsion.k[j] / energy_unit
            torsion_phases[i, p - 1] = torsion.phase[j] / angle_unit
    return np.array(torsion_inds), torsion_ks, torsion_phases

# impropers
# TODO: include impropers


# all together
def form_valence_targets(labeled_mol):
    inds, valence_targets = dict(), dict()

    # atoms
    # TODO: include LJ

    # bonds
    inds['bonds'], length, bond_k = extract_bond_params(labeled_mol)
    valence_targets['bonds'] = torch.Tensor(np.vstack((length, bond_k)).T)

    # angles
    inds['angles'], eq_angles, angle_k = extract_angle_params(labeled_mol)
    valence_targets['angles'] = torch.Tensor(np.vstack((eq_angles, angle_k)).T)

    # propers
    inds['torsions'], ks, phases = extract_torsion_params(labeled_mol)
    valence_targets['torsions'] = torch.Tensor(np.hstack((ks, phases)))

    # impropers
    # TODO: include impropers

    return inds, valence_targets
