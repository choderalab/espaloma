# evaluate ani energies and forces on saved snapshots
# TODO: maybe split up into per-atom and per-net contributions...


import mdtraj as md
import numpy as onp
import torch
import torchani
from pkg_resources import resource_filename
from simtk import unit
from torchani.units import hartree2kjoulemol

energy_unit = unit.kilojoule_per_mole
force_unit = unit.kilojoule_per_mole / unit.nanometer


def _compute_ani(snapshots, return_energies=True, return_forces=False):
    """return a dict with optional keys 'energies' and 'forces' """
    xyz_in_angstroms = (snapshots.xyz * unit.nanometer).value_in_unit(unit.angstrom)
    species_string = ''.join([a.element.symbol for a in snapshots.topology.atoms])

    species = model.species_to_tensor(species_string).unsqueeze(0)

    coordinates = torch.tensor([sample for sample in xyz_in_angstroms], dtype=torch.float32, requires_grad=True)

    _, energy = model((torch.stack([species[0]] * len(xyz_in_angstroms)), coordinates))  # hartree

    return_dict = {}
    if return_energies:
        return_dict['energies'] = hartree2kjoulemol(energy.detach().numpy()) * unit.kilojoule_per_mole
    if return_forces:
        forces = -torch.autograd.grad(energy.sum(), coordinates, create_graph=True, retain_graph=True)[
            0]  # hartree per angstrom
        return_dict['forces'] = hartree2kjoulemol(forces.detach().numpy()) * unit.kilojoule_per_mole / unit.angstrom

    return return_dict


def compute_ani_energies(snapshots: md.Trajectory):
    """return unit'd arrray of energies in kJ/mol"""
    return _compute_ani(snapshots, return_energies=True)['energies']


def compute_ani_forces(snapshots: md.Trajectory):
    """return unit'd array of forces in kJ/mol / angstrom"""
    return _compute_ani(snapshots, return_forces=True)['forces']


def compute_ani_energies_and_forces(snapshots: md.Trajectory):
    result_dict = _compute_ani(snapshots, return_energies=True, return_forces=True)
    return result_dict['energies'], result_dict['forces']


def get_snapshots_and_energies(name='AlkEthOH_r1155'):
    snapshots_path = 'snapshots_and_energies/{}_molecule_traj.h5'.format(name)
    energies_path = 'snapshots_and_energies/{}_molecule_energies.npy'.format(name)

    snapshots = md.load(snapshots_path)
    energies = onp.load(energies_path)

    return snapshots, energies


def get_snapshots_energies_and_forces(name='AlkEthOH_r1155'):
    snapshots_path = resource_filename('espaloma.data.alkethoh',
                                       'snapshots_and_energies/{}_molecule_traj.h5'.format(name))
    ani1ccx_energies_path = resource_filename('espaloma.data.alkethoh',
                                              'snapshots_and_energies/{}_ani1ccx_energies.npy'.format(name))
    ani1ccx_forces_path = resource_filename('espaloma.data.alkethoh',
                                            'snapshots_and_energies/{}_ani1ccx_forces.npy'.format(name))

    snapshots = md.load(snapshots_path)
    ani1ccx_energies = onp.load(ani1ccx_energies_path)
    ani1ccx_forces = onp.load(ani1ccx_forces_path)

    # double-check shapes
    assert (len(ani1ccx_energies.shape) == 1)
    assert (len(ani1ccx_forces.shape) == 3)
    assert (ani1ccx_forces.shape == snapshots.xyz.shape)

    return snapshots, ani1ccx_energies, ani1ccx_forces


model = torchani.models.ANI1ccx()

if __name__ == '__main__':
    from pickle import load
    from tqdm import tqdm
    from espaloma.data.alkethoh.data import offmols

    names = list(offmols.keys())

    for name in tqdm(names):
        snapshots, energies = get_snapshots_and_energies(name)
        ani_energies, ani_forces = compute_ani_energies_and_forces(snapshots)

        # save energies
        ani_energies_in_omm_units = ani_energies.value_in_unit(energy_unit)
        onp.save('snapshots_and_energies/{}_ani1ccx_energies'.format(name), ani_energies_in_omm_units)

        # save forces
        ani_forces_in_omm_units = ani_forces / force_unit
        onp.save('snapshots_and_energies/{}_ani1ccx_forces'.format(name), ani_forces_in_omm_units)

        # print openff vs. ani residual stddev
        mm_energies_kjmol = (energies * unit.kilojoule_per_mole).value_in_unit(unit.kilojoule_per_mole)
        residuals = ani_energies_in_omm_units - mm_energies_kjmol
        print('stddev(residuals): {:.4f} kJ/mol'.format(onp.std(residuals)))
