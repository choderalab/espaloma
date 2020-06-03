# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import torch
from espaloma.data.dataset import MoleculeDataset

# =============================================================================
# MODULE CLASSES
# =============================================================================
class ESOL(MoleculeDataset):
    def __init__(self):
        import pandas as pd
        import os
        from openforcefield.topology import Molecule
        path = os.path.dirname(esp.__file__) + '/data/esol.csv'
        df = pd.read_csv(path)
        smiles = df.iloc[:, -1]
        mols = [Molecule.from_smiles(
        _smiles, allow_undefined_stereo=True) for _smiles in smiles]
        
        super(ESOL, self).__init__(mols)



