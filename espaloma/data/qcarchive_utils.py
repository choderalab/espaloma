# =============================================================================
# IMPORTS
# =============================================================================
from collections import namedtuple
from typing import Tuple

import numpy as np
import qcportal as ptl
import torch
from openmm import unit
from openmm.unit import Quantity

import espaloma as esp


# =============================================================================
# CONSTANTS
# =============================================================================


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_client():
    return ptl.PortalClient("api.qcarchive.molssi.org")


def get_collection(
        client,
        collection_type="OptimizationDataset",
        name="OpenFF Full Optimization Benchmark 1",
):
    collection = client.get_dataset(
        dataset_type=collection_type,
        dataset_name=name,
    )

    record_names = collection.entry_names

    return collection, record_names


def get_graph(collection, record_name):
    # get record and trajectory
    record = collection.get_record(record_name, specification_name="default")
    entry = collection.get_entry(record_name)
    from openff.toolkit.topology import Molecule

    mol = Molecule.from_qcschema(entry)

    try:
        trajectory = record.trajectory
    except:
        return None

    if trajectory is None:
        return None

    g = esp.Graph(mol)

    # energy is already hartree
    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                snapshot.properties["scf_total_energy"],
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
            for snapshot in trajectory
        ],
        dtype=torch.get_default_dtype(),
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    snapshot.molecule.geometry,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
                for snapshot in trajectory
            ],
            axis=1,
        ),
        requires_grad=True,
        dtype=torch.get_default_dtype(),
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.stack(
        [
            torch.tensor(
                Quantity(
                    np.array(snapshot.properties["return_result"]).reshape((-1, 3)),
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                dtype=torch.get_default_dtype(),
            )
            for snapshot in trajectory
        ],
        dim=1,
    )

    return g


def fetch_td_record(record: ptl.torsiondrive.record_models.TorsiondriveRecord):
    molecule_optimization = record.optimizations

    angle_keys = list(molecule_optimization.keys())

    xyzs = []
    energies = []
    gradients = []

    for angle in angle_keys:
        # NOTE: this is calling the first index of the optimization array
        # this gives the same value as the prior implementation, but I wonder if it
        # should be molecule_optimization[angle][-1] in both cases
        mol = molecule_optimization[angle][0].final_molecule
        result = molecule_optimization[angle][0].trajectory[-1].properties

        """Note: force = - gradient"""

        # TODO: attach units here? or later?

        e = result["current energy"]
        g = np.array(result["current gradient"]).reshape(-1, 3)

        xyzs.append(mol.geometry)
        energies.append(e)
        gradients.append(g)

    # to arrays
    xyz = np.array(xyzs)
    energies = np.array(energies)
    gradients = np.array(gradients)

    # assume each angle key is a tuple -- sort by first angle in tuple

    # NOTE: (for now making the assumption that these torsion drives are 1D)
    for k in angle_keys:
        assert len(k) == 1

    to_ordered = np.argsort([k[0] for k in angle_keys])
    angles_in_order = [angle_keys[i_] for i_ in to_ordered]
    flat_angles = np.array(angles_in_order).flatten()

    # put the xyz's, energies, and gradients in the same order as the angles
    xyz_in_order = xyz[to_ordered]
    energies_in_order = energies[to_ordered]
    gradients_in_order = gradients[to_ordered]

    # TODO: put this return blob into a better struct
    return flat_angles, xyz_in_order, energies_in_order, gradients_in_order


MolWithTargets = namedtuple(
    "MolWithTargets", ["offmol", "xyz", "energies", "gradients"]
)


def h5_to_dataset(df):
    def get_smiles(x):
        try:
            return x["offmol"].to_smiles()
        except:
            return np.nan

    df["smiles"] = df.apply(get_smiles, axis=1)
    df = df.dropna()
    groups = df.groupby("smiles")
    gs = []
    for name, group in groups:
        mol_ref = group["offmol"][0]
        assert all(mol_ref == entry for entry in group["offmol"])
        g = esp.Graph(mol_ref)

        u_ref = np.concatenate(group["energies"].values)
        u_ref_prime = np.concatenate(group["gradients"].values, axis=0).transpose(
            1, 0, 2
        )
        xyz = np.concatenate(group["xyz"].values, axis=0).transpose(1, 0, 2)

        assert u_ref_prime.shape[0] == xyz.shape[0] == mol_ref.n_atoms
        assert u_ref.shape[0] == u_ref_prime.shape[1] == xyz.shape[1]

        # energy is already hartree
        g.nodes["g"].data["u_ref"] = torch.tensor(
            Quantity(u_ref, esp.units.HARTREE_PER_PARTICLE).value_in_unit(
                esp.units.ENERGY_UNIT
            ),
            dtype=torch.get_default_dtype(),
        )[None, :]

        g.nodes["n1"].data["xyz"] = torch.tensor(
            Quantity(
                xyz,
                unit.bohr,
            ).value_in_unit(esp.units.DISTANCE_UNIT),
            requires_grad=True,
            dtype=torch.get_default_dtype(),
        )

        g.nodes["n1"].data["u_ref_prime"] = torch.tensor(
            Quantity(
                u_ref_prime,
                esp.units.HARTREE_PER_PARTICLE / unit.bohr,
            ).value_in_unit(esp.units.FORCE_UNIT),
            dtype=torch.get_default_dtype(),
        )

        gs.append(g)

    return esp.data.dataset.GraphDataset(gs)


def breakdown_along_time_axis(g, batch_size=32):
    n_snapshots = g.nodes["g"].data["u_ref"].flatten().shape[0]
    idxs = list(range(n_snapshots))
    from random import shuffle

    shuffle(idxs)
    chunks = [
        idxs[_idx * batch_size: (_idx + 1) * batch_size]
        for _idx in range(n_snapshots // batch_size)
    ]

    _gs = []
    for chunk in chunks:
        _g = esp.Graph(g.mol)
        _g.nodes["g"].data["u_ref"] = (
            g.nodes["g"].data["u_ref"][:, chunk].detach().clone()
        )
        _g.nodes["n1"].data["xyz"] = (
            g.nodes["n1"].data["xyz"][:, chunk, :].detach().clone()
        )
        _g.nodes["n1"].data["u_ref_prime"] = (
            g.nodes["n1"].data["u_ref_prime"][:, chunk, :].detach().clone()
        )

        _g.nodes["n1"].data["xyz"].requires_grad = True

        _gs.append(_g)

    return _gs


def make_batch_size_consistent(ds, batch_size=32):
    import itertools

    return esp.data.dataset.GraphDataset(
        list(
            itertools.chain.from_iterable(
                [breakdown_along_time_axis(g, batch_size=batch_size) for g in ds]
            )
        )
    )


def weight_by_snapshots(g, key="weight"):
    n_snapshots = g.nodes["n1"].data["xyz"].shape[1]
    g.nodes["g"].data[key] = torch.tensor(float(1.0 / n_snapshots))[None, :]
