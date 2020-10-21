# loop over all the pkl files in a folder, each containing a dict whose 3 values are also dictionaries
#    result = dict(
#        mols_and_targets=mols_and_targets,
#        exceptions=exceptions,
#        skipped_status=skipped_status
#    )

from glob import glob
from pickle import load, dump
from typing import List, Dict

from espaloma.data.qcarchive_utils import MolWithTargets


def fetch_dictionaries(path: str) -> List[dict]:
    paths = glob(path + '*.pkl')
    print('fetching pkl files from the following paths: ', paths)
    results = []

    for path in paths:
        with open(path, 'rb') as f:
            results.append(load(f))
    return results


def merge_dictionaries(dictionaries: List[dict]) -> dict:
    merged = dict()

    for d in dictionaries:
        merged.update(d)

    return merged


if __name__ == '__main__':

    import sys

    name = sys.argv[1]

    path = f'batches/{name}/'

    results = fetch_dictionaries(path)

    merged_results = dict()
    keys = list(results[0].keys())

    print('merging...')
    for key in keys:
        merged_results[key] = merge_dictionaries([r[key] for r in results])
        print(f'# of total entries for key "{key}": {len(merged_results[key])}')

    # annotate type of mols_and_targets dict
    merged_results['mols_and_targets']: Dict[str, MolWithTargets]

    # save merged result
    with open(f'{name}.pkl', 'wb') as f:
        dump(merged_results, f)
