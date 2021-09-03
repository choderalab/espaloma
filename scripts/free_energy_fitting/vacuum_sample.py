from simtk import unit
from simtk.openmm import app

distance_unit = unit.nanometer
from simtk.openmm import XmlSerializer
from tqdm import tqdm
from openmmtools.integrators import BAOABIntegrator

import numpy as np
import pandas as pd

df = pd.read_hdf('freesolv.h5')

keys = sorted(list(df.index))

temperature = 300 * unit.kelvin
gamma = 1.0 / unit.picosecond
dt = 1.0 * unit.femtosecond


def get_vacuum_sim(key):
    """load topology and serialized vacuum simulation"""
    offmol = df['offmol'][key]

    top = offmol.to_topology().to_openmm()
    sys = XmlSerializer.deserializeSystem(df['serialized_openmm_system'][key])

    sim = app.Simulation(top, sys, BAOABIntegrator(temperature, gamma, dt))

    if offmol.conformers is None:
        offmol.generate_conformers()

    sim.context.setPositions(offmol.conformers[0])
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature)

    return sim


def generate_trajectory(sim, burn_in=1000, n_samples=100, thinning=1000):
    def get_positions():
        return sim.context.getState(getPositions=True).getPositions(
            asNumpy=True)

    sim.step(burn_in)
    xyz = []
    for _ in tqdm(range(n_samples)):
        sim.step(thinning)
        xyz.append(get_positions().value_in_unit(distance_unit))
    return np.array(xyz)


if __name__ == '__main__':
    import sys

    batch_index = int(sys.argv[1])
    key = keys[batch_index]
    xyz = generate_trajectory(get_vacuum_sim(key), burn_in=10000,
                              n_samples=1000, thinning=10000)
    np.save(f'parsley12_vacuum_samples/{key}.npy', xyz)
