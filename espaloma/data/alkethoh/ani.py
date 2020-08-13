# evaluate ani energies on saved snapshots
# TODO: also forces, and maybe split up into per-atom and per-net contributions...


import mdtraj as md
import numpy as np
import torch
import torchani
from simtk import unit
from torchani.units import hartree2kcalmol

energy_unit = unit.kilojoule_per_mole
force_unit = unit.kilojoule_per_mole / unit.nanometer


def get_snapshots_and_energies(name='AlkEthOH_r1155'):
    snapshots_path = 'snapshots_and_energies/{}_molecule_traj.h5'.format(name)
    energies_path = 'snapshots_and_energies/{}_molecule_energies.npy'.format(name)

    snapshots = md.load(snapshots_path)
    energies = np.load(energies_path)

    return snapshots, energies


def compute_ani_energies(snapshots: md.Trajectory):
    xyz_in_angstroms = (snapshots.xyz * unit.nanometer).value_in_unit(unit.angstrom)
    species_string = ''.join([a.element.symbol for a in snapshots.topology.atoms])

    species = model.species_to_tensor(species_string).unsqueeze(0)

    coordinates = torch.tensor([sample for sample in xyz_in_angstroms], dtype=torch.float32)
    _, energy = model((torch.stack([species[0]] * len(xyz_in_angstroms)), coordinates))  # hartree
    return hartree2kcalmol(energy.detach().numpy()) * unit.kilocalorie_per_mole


def compute_ani_forces(snapshots: md.Trajectory):
    xyz_in_angstroms = (snapshots.xyz * unit.nanometer).value_in_unit(unit.angstrom)
    species_string = ''.join([a.element.symbol for a in snapshots.topology.atoms])

    species = model.species_to_tensor(species_string).unsqueeze(0)

    coordinates = torch.tensor([sample for sample in xyz_in_angstroms], dtype=torch.float32, requires_grad=True)

    _, energy = model((torch.stack([species[0]] * len(xyz_in_angstroms)), coordinates))  # hartree

    forces = -torch.autograd.grad(energy.sum(), coordinates, create_graph=True, retain_graph=True)[
        0]  # hartree per angstrom

    return hartree2kcalmol(forces.detach().numpy()) * unit.kilocalorie_per_mole / unit.angstrom


def compute_ani_energies_and_forces(snapshots: md.Trajectory):
    xyz_in_angstroms = (snapshots.xyz * unit.nanometer).value_in_unit(unit.angstrom)
    species_string = ''.join([a.element.symbol for a in snapshots.topology.atoms])

    species = model.species_to_tensor(species_string).unsqueeze(0)

    coordinates = torch.tensor([sample for sample in xyz_in_angstroms], dtype=torch.float32, requires_grad=True)

    _, energy = model((torch.stack([species[0]] * len(xyz_in_angstroms)), coordinates))  # hartree

    forces = -torch.autograd.grad(energy.sum(), coordinates, create_graph=True, retain_graph=True)[
        0]  # hartree per angstrom

    energies = hartree2kcalmol(energy.detach().numpy()) * unit.kilocalorie_per_mole
    forces = hartree2kcalmol(forces.detach().numpy()) * unit.kilocalorie_per_mole / unit.angstrom
    return energies, forces


model = torchani.models.ANI1ccx()

if __name__ == '__main__':
    from pickle import load
    from tqdm import tqdm

    with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
        mols = load(f)

    names = list(mols.keys())

    for name in tqdm(names):
        snapshots, energies = get_snapshots_and_energies(name)
        ani_energies, ani_forces = compute_ani_energies_and_forces(snapshots)

        # save energies
        ani_energies_in_omm_units = ani_energies.value_in_unit(energy_unit)
        np.save('snapshots_and_energies/{}_ani1ccx_energies'.format(name), ani_energies_in_omm_units)

        # save forces
        ani_forces_in_omm_units = ani_forces / force_unit
        np.save('snapshots_and_energies/{}_ani1ccx_forces'.format(name), ani_forces_in_omm_units)

        # print openff vs. ani residual stddev
        mm_energies_kjmol = (energies * unit.kilojoule_per_mole).value_in_unit(unit.kilojoule_per_mole)
        residuals = ani_energies_in_omm_units - mm_energies_kjmol
        print('stddev(residuals): {:.4f} kJ/mol'.format(np.std(residuals)))
