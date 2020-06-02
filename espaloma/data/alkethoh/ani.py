# evaluate ani energies on saved snapshots
# TODO: also forces, and maybe split up into per-atom and per-net contributions...


import mdtraj as md
import numpy as np
import torch
import torchani
from simtk import unit


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
    energy = model((torch.stack([species[0]] * len(xyz_in_angstroms)), coordinates))
    return energy.energies.detach().numpy() * 627.5 * unit.kilocalorie_per_mole  # convert from hartree to kcal/mol


model = torchani.models.ANI1ccx()

if __name__ == '__main__':
    from pickle import load
    from tqdm import tqdm

    with open('AlkEthOH_rings_offmols.pkl', 'rb') as f:
        mols = load(f)

    names = list(mols.keys())

    for name in tqdm(names):
        snapshots, energies = get_snapshots_and_energies(name)
        ani_energies = compute_ani_energies(snapshots)
        ani_energies_kjmol = ani_energies.value_in_unit(unit.kilojoule_per_mole)
        np.save('snapshots_and_energies/{}_ani1ccx_energies.npz'.format(name), ani_energies_kjmol)

        # note: these will include a large additive offset: interested in stddev of these residuals
        mm_energies_kjmol = (energies * unit.kilojoule_per_mole).value_in_unit(unit.kilojoule_per_mole)
        residuals = ani_energies_kjmol - mm_energies_kjmol
        print('stddev(residuals): {:.4f} kJ/mol'.format(np.std(residuals)))
