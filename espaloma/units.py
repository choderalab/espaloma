# =============================================================================
# IMPORTS
# =============================================================================
from simtk import unit

# =============================================================================
# CONSTANTS
# =============================================================================

# scaled units
PARTICLE = unit.mole.create_unit(
    6.02214076e23 ** -1,
    'particle',
    'particle',
)


HARTREE_PER_PARTICLE = unit.hartree / PARTICLE

# basic units
DISTANCE_UNIT = unit.bohr
ENERGY_UNIT = HARTREE_PER_PARTICLE
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT
ANGLE_UNIT = unit.radian
CHARGE_UNIT = unit.elementary_charge

# compose units
FORCE_CONSTANT_UNIT = ENERGY_UNIT / (DISTANCE_UNIT ** 2)
ANGLE_FORCE_CONSTANCE_UNIT = ENERGY_UNIT / (ANGLE_UNIT ** 2)
# COULOMB_CONSTANT_UNIT = ENERGY_UNIT * DISTANCE_UNIT / (
#     unit.mole * (unit.elementary_charge ** 2))
