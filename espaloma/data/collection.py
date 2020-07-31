# =============================================================================
# IMPORTS
# =============================================================================
import abc

import torch

import espaloma as esp
from espaloma.data.dataset import GraphDataset


# =============================================================================
# MODULE CLASSES
# =============================================================================
def esol(*args, **kwargs):
    import os

    import pandas as pd
    from openforcefield.topology import Molecule

    path = os.path.dirname(esp.__file__) + "/data/esol.csv"
    df = pd.read_csv(path)
    smiles = df.iloc[:, -1]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)


def alkethoh(*args, **kwargs):
    import os

    import pandas as pd
    from openforcefield.topology import Molecule

    path = os.path.dirname(esp.__file__) + "/data/alkethoh.smi"
    df = pd.read_csv(path)
    smiles = df.iloc[:, 0]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)
