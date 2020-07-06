# =============================================================================
# IMPORTS
# =============================================================================
from simtk import unit

# =============================================================================
# CONSTANTS
# =============================================================================
# basic units
DISTANCE_UNIT = unit.nanometer
ENERGY_UNIT = unit.kilojoule_per_mole
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT
ANGLE_UNIT = unit.radian

# compose units
FORCE_CONSTANT_UNIT = ENERGY_UNIT / (DISTANCE_UNIT ** 2)
ANGLE_FORCE_CONSTANCE_UNIT = ENERGY_UNIT / (ANGLE_UNIT ** 2)
# COULOMB_CONSTANT_UNIT = ENERGY_UNIT * DISTANCE_UNIT / (
#     unit.mole * (unit.elementary_charge ** 2))
