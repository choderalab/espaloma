import io
import logging
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO  # Python 3
from multiprocessing import Process

import rdkit.Chem.rdForceFieldHelpers as ff
from rdkit import Chem
from rdkit.Chem import AllChem

if __name__ == '__main__':
    smile = sys.argv[1]
    m = Chem.MolFromSmiles(smile)

    m2 = Chem.AddHs(m)
    AllChem.EmbedMolecule(m2)
    AllChem.MMFFOptimizeMolecule(m2)
    pr = ff.MMFFGetMoleculeProperties(m2)

    pr.SetMMFFAngleTerm(False)
    pr.SetMMFFBondTerm(False)
    pr.SetMMFFOopTerm(False)
    pr.SetMMFFStretchBendTerm(False)
    pr.SetMMFFTorsionTerm(False)

    pr.SetMMFFVerbosity(2)

    mmff = ff.MMFFGetMoleculeForceField(m2,pr)
    print(mmff.CalcEnergy())
