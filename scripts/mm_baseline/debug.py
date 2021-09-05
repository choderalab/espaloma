import qcportal as ptl
from simtk import unit
from openmmforcefields.generators import SystemGenerator
from simtk import openmm, unit
from simtk.openmm.app import Simulation
from simtk.unit import Quantity

from espaloma.units import *
import espaloma as esp

# =============================================================================
# CONSTANTS
# =============================================================================
# simulation specs
TEMPERATURE = 350 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond
EPSILON_MIN = 0.05 * unit.kilojoules_per_mole

def run():
    # scaled units
    PARTICLE = unit.mole.create_unit(6.02214076e23 ** -1, "particle", "particle",)

    HARTREE_PER_PARTICLE = unit.hartree / PARTICLE

    # example record
    record_name = "c1cc2c(c(c[nh]2)c[c@@h](co)[nh3+])nc1-0"

    # get record
    client = ptl.FractalClient()
    collection = client.get_collection(
        "OptimizationDataset",
        "OpenFF Gen 2 Opt Set 5 Bayer",
    )
    record = collection.get_record(record_name, specification="default")
    entry = collection.get_entry(record_name)
    from openforcefield.topology import Molecule
    mol = Molecule.from_qcschema(entry)
    trajectory = record.get_trajectory()

    import numpy as np
    from simtk.unit.quantity import Quantity

    u_qm = np.array(
        [
                Quantity(
                    snapshot.properties.scf_total_energy,
                    HARTREE_PER_PARTICLE
                ).value_in_unit(
                    unit.kilocalories_per_mole
                )
                for snapshot in trajectory
        ]
    )

    xs = np.stack(
                [
                    Quantity(
                        snapshot.get_molecule().geometry,
                        unit.bohr,
                    ).value_in_unit(
                        unit.nanometer
                    )
                    for snapshot in trajectory
                ],
                axis=0
    )


    # define a system generator
    system_generator = SystemGenerator(
        small_molecule_forcefield="openff-1.2.0",
    )

    # mol.assign_partial_charges("formal_charge")
    # create system
    system = system_generator.create_system(
        topology=mol.to_topology().to_openmm(),
        molecules=mol,
    )

    # parameterize topology
    topology = g.mol.to_topology().to_openmm()

    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE
    )

    # create simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator
    )

    u_mm = []

    for x in xs:
        simulation.context.setPositions(x)
        u_mm.append(
            simulation.context.getState(
                getEnergy=True
            ).getPotentialEnergy().value_in_unit(
                esp.units.ENERGY_UNIT
            )
        )


if __name__ == "__main__":
    run()
