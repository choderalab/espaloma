import qcportal as ptl
from simtk import unit


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

    xyz = np.stack(
                [
                    Quantity(
                        snapshot.get_molecule().geometry,
                        unit.bohr,
                    ).value_in_unit(
                        unit.nanometer
                    )
                    for snapshot in trajectory
                ],
                axis=1
    )
if __name__ == "__main__":
    run()
