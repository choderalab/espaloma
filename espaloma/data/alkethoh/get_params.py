from openforcefield.typing.engines.smirnoff import ForceField
from tqdm import tqdm
from openforcefield.typing.engines.smirnoff import parameters
ProperTorsionType = parameters.ProperTorsionHandler.ProperTorsionType

forcefield = ForceField('openff_unconstrained-1.0.0.offxml')


from simtk import unit
import numpy as np


class FlatTorsion():
    """periodicities fixed at n=1..6,
    but variable ks and phases
    """
    periodicities = np.arange(6) + 1
    k_unit = unit.kilojoule_per_mole
    phase_unit = unit.radian

    def __init__(self, torsion: ProperTorsionType):
        self.ks = np.zeros(6)
        self.phases = np.zeros(6)

        for (n, k, phase) in zip(torsion.periodicity, torsion.k, torsion.phase):
            ind = n - 1  # python zero-indexing
            self.ks[ind] = k / self.k_unit
            self.phases[ind] = phase / self.phase_unit

    def __repr__(self):
        return f"Torsion(ks={self.ks}; phases={self.phases})"

