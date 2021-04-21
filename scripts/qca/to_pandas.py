# load a merged dictionary (saved by merge_batches.py), save to a pandas dataframe with QCA record names as index,
# and [offmol, xyz, energies, and gradients] as columns


from pickle import load
from typing import Dict
from espaloma.data.qcarchive_utils import MolWithTargets

import pandas as pd


if __name__ == '__main__':
    import sys

    name = sys.argv[1]

    with open(f'{name}.pkl', 'rb') as f:
        pkl = load(f)

    mols_and_targets: Dict[str, MolWithTargets] = pkl['mols_and_targets']

    index = list(mols_and_targets.keys())

    df = pd.DataFrame(index=index, columns=['offmol', 'xyz', 'energies', 'gradients'])

    for key in index:
        m = mols_and_targets[key]

        df.offmol[key] = m.offmol
        df.xyz[key] = m.xyz
        df.energies[key] = m.energies
        df.gradients[key] = m.gradients

    print(df.columns)
    print(df)

    # TODO: possibly address this performance warning
    #   PerformanceWarning:
    #    your performance may suffer as PyTables will pickle object types that it cannot
    #    map directly to c-types [inferred_type->mixed,key->block0_values]
    #    [items->Index(['offmol', 'xyz', 'energies', 'gradients'], dtype='object')]
    df.to_hdf(f'{name}.h5', key='df')