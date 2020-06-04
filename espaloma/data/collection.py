# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import torch
from espaloma.data.dataset import GraphDataset

# =============================================================================
# MODULE CLASSES
# =============================================================================
def esol(*args, **kwargs):
    import pandas as pd
    import os
    from openforcefield.topology import Molecule
    path = os.path.dirname(esp.__file__) + '/data/esol.csv'
    df = pd.read_csv(path)
    smiles = df.iloc[:, -1]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)



           


