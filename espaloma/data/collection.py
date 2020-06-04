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
class ESOL(GraphDataset):
    def __init__(self, *args, **kwargs):
        import pandas as pd
        import os
        from openforcefield.topology import Molecule
        path = os.path.dirname(esp.__file__) + '/data/esol.csv'
        df = pd.read_csv(path)
        smiles = df.iloc[:, -1]
        
        super(ESOL, self).__init__(smiles, *args, **kwargs)



