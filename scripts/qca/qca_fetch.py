# Attempt to download and save Pandas dataframes for all `OptimizationDataset`s, `TorsionDriveDataset`s, and `GridOptimizationDataset`s in QCArchive

from collections import namedtuple

import numpy as np
import qcportal
from dataset_selection import optimization_datasets, dataset_type
from openforcefield.topology import Molecule
from tqdm import tqdm
from pickle import dump

# Initialize FractalClient
# As documented here: http://docs.qcarchive.molssi.org/projects/QCPortal/en/stable/client.html
from espaloma.data.qcarchive_utils import get_energy_and_gradient

client = qcportal.FractalClient()

MolWithTargets = namedtuple('MolWithTargets', ['offmol', 'xyz', 'energies', 'gradients'])


def get_mol_with_targets(record, entry) -> MolWithTargets:
    # offmol
    offmol = Molecule.from_qcschema(entry)

    # trajectory containing xyz, energies, and gradients
    trajectory = record.get_trajectory()

    # xyz
    molecules = [snapshot.get_molecule() for snapshot in trajectory]
    xyz = np.array([mol.geometry for mol in molecules])

    # energies and gradients
    energies_and_gradients = list(map(get_energy_and_gradient, trajectory))
    energies = np.array([e for (e, _) in energies_and_gradients])
    gradients = np.array([g for (_, g) in energies_and_gradients])

    return MolWithTargets(offmol, xyz, energies, gradients)


def fetch(ds, keys, specification='default'):
    mols_and_targets = dict()
    exceptions = dict()
    skipped_status = dict()

    trange = tqdm(keys)
    for key in trange:

        # record and entry
        record = ds.get_record(key, specification)
        entry = ds.get_entry(key)

        # offmol with targets
        if (record.status == 'COMPLETE') and (key not in mols_and_targets):
            try:
                mols_and_targets[key] = get_mol_with_targets(record, entry)
            except Exception as e:
                print(f'unspecified problem encountered with {key}!')
                exceptions[key] = e
        elif (key not in mols_and_targets):
            print(f'skipping {key}, which has a status {record.status}')
            skipped_status[key] = record.status

        trange.set_postfix(
            n_skipped=len(skipped_status),
            n_exceptions=len(exceptions),
            n_successful=len(mols_and_targets)
        )

    return mols_and_targets, exceptions, skipped_status


if __name__ == '__main__':
    # get arguments
    import sys

    short_name = sys.argv[1]
    batch_index = int(sys.argv[2])

    print(short_name, batch_index)
    path_to_keys = f'batches/{short_name}/{batch_index}.txt'
    out_path = f'batches/{short_name}/{batch_index}.pkl'
    print(f'reading from {path_to_keys}')
    print(f'writing to {out_path}')

    dataset_name = optimization_datasets[short_name]
    ds = client.get_collection(dataset_type, dataset_name)

    with open(path_to_keys, 'r') as f:
        keys = [s.strip() for s in f.readlines()]
    print(keys)

    mols_and_targets, exceptions, skipped_status = fetch(ds, keys)
    result = dict(
        mols_and_targets=mols_and_targets,
        exceptions=exceptions,
        skipped_status=skipped_status
    )

    with open(out_path, 'wb') as f:
        dump(result, f)
