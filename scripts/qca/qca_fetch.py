# Attempt to download and save Pandas dataframes for all `OptimizationDataset`s, `TorsionDriveDataset`s, and `GridOptimizationDataset`s in QCArchive

from collections import namedtuple
from pickle import dump
from typing import Tuple

import numpy as np
import qcportal
from openforcefield.topology import Molecule
from tqdm import tqdm

# Initialize FractalClient
# As documented here: http://docs.qcarchive.molssi.org/projects/QCPortal/en/stable/client.html

client = qcportal.FractalClient()

# print all collections
collections: dict = client.list_collections(aslist=True)
for dataset_type in collections.keys():
    dataset_names = sorted(collections[dataset_type])
    print(f'"{dataset_type}" ({len(dataset_names)} datasets)')
    for i, dataset_name in enumerate(dataset_names):
        print(f'\t{i}: "{dataset_name}"')

# TODO: Loop over `(dataset_type, dataset_name)` pairs

# Fetch dataset

dataset_type = 'OptimizationDataset'
dataset_name = 'OpenFF Full Optimization Benchmark 1'

ds = client.get_collection(dataset_type, dataset_name)
print('dir(ds)', dir(ds))
print('type(ds)', type(ds))
print('len(ds.df)', len(ds.df))
print('ds.list_specifications()', ds.list_specifications())

specifications = ds.list_specifications(description=False)

# TODO: Also loop over specifications, in case there's more than one
specification = specifications[0]

index = list(ds.df.index)
record_names = list(ds.data.records)


def get_energy_and_gradient(snapshot: qcportal.models.records.ResultRecord) -> Tuple[float, np.ndarray]:
    """Note: force = - gradient"""
    d = snapshot.dict()
    qcvars = d['extras']['qcvars']
    energy = qcvars['CURRENT ENERGY']
    flat_gradient = np.array(qcvars['CURRENT GRADIENT'])
    num_atoms = len(flat_gradient) // 3
    gradient = flat_gradient.reshape((num_atoms, 3))
    return energy, gradient


MolWithTargets = namedtuple('MolWithTargets', ['offmol', 'xyz', 'energies', 'gradients'])


def get_mol_with_targets(record:, entry) -> MolWithTargets:
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


# fetch everything
all_mols_and_targets = dict()
exceptions = dict()
skipped_status = dict()

trange = tqdm(range(len(index)))
for i in trange:

    # record and entry
    record_name = record_names[i]
    ind = index[i]

    record = ds.get_record(record_name, specification)
    entry = ds.get_entry(ind)

    # offmol with targets
    status = record.status
    status_string = status.name

    if status_string == 'complete':
        try:
            all_mols_and_targets[record_name] = get_mol_with_targets(record, entry)
        except Exception as e:
            print(f'unspecified problem encountered with {record_name}!')
            exceptions[record_name] = e
    else:
        print(f'skipping {record_name}, which has a status {status}')
        skipped_status[record_name] = status

    trange.set_postfix(
        n_skipped=len(skipped_status),
        n_exceptions=len(exceptions),
        n_successful=len(all_mols_and_targets)
    )

# what was the record status enum for each of the records we skipped?
print('set(skipped_status.values())', set(skipped_status.values()))

# save what we have for now...

# TODO: does this ds.df thing contain all of the trajectories also?
ds.df.to_hdf('some_of_optimization_dataset.h5', key='df')

with open('all_mols_and_targets.pkl', 'wb') as f:
    dump(all_mols_and_targets, f)

# TODO: allow batches
