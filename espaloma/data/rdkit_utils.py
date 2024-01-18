import io
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO  # Python 3
from multiprocessing import Process
from typing import Dict, List

import rdkit
import rdkit.Chem.rdForceFieldHelpers as ff
from openmm import unit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

FORCES = ["VANDERWAALS", "ELECTROSTATIC"]
END = "TOTAL"
COL_KEYWORD = "ENERGY"
ERR_TOLLERANCE = 0.001
RDKIT_ENERGY_UNIT = unit.kilocalories_per_mole

RDKIT_FORCE_UNIT = unit.Unit({unit.BaseUnit(base_dim=unit.BaseDimension("length"), name="nanometer", symbol="nm"): -1.0, unit.BaseUnit(base_dim=unit.BaseDimension("amount"), name="mole", symbol="mol"): -1.0, unit.ScaledUnit(factor=1.0, master=unit.gram*unit.nanometer**2/(unit.picosecond**2), name='kilojoule', symbol='kJ'): 1.0})


def _parse_rdkit_output(rdkit_cout: str) -> Dict[str, List[str]]:
    """
    Parse RDKIT HIGH_VERBOSITY output to command line and return
    dictionary of rows per energy.

    E.g.,
    'V A N   D E R   W A A L S\n',
     '\n',
     '------ATOMS------   ATOM TYPES                                 WELL\n',
     '  I        J          I    J    DISTANCE   ENERGY     R*      DEPTH\n',
     '--------------------------------------------------------------------\n',
     'O  #1    C  #4        6   63      3.752    -0.056    3.936    0.063\n',
     'O  #1    N  #7        6   81      5.902    -0.006    3.621    0.074\n',
    ...

    Returns {"VANDERWAALS": [
    'O  #1    C  #4        6   63      3.752    -0.056    3.936    0.063\n',
     'O  #1    N  #7        6   81      5.902    -0.006    3.621    0.074\n',
    ]}

    """
    parse = defaultdict(list)
    collect = False
    cur_energy = ""
    
    for r in rdkit_cout:
        if not r:
            continue
        if r.replace(" ", "") in FORCES:
            cur_energy = r
            collect = True
        elif END in r:
            collect = False
        elif collect:
            parse[cur_energy].append(r)
    return parse


def _get_rdkit_cout(smile: str) -> str:


    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'get_properties.py'),  smile], capture_output=True, text=True)
    if not out.stdout:
        raise Exception(out.stderr)


    return out.stdout


def get_energy_contributions(mol )-> Dict[int, float]:
    """
    E.g.,  
    'V A N   D E R   W A A L S\n',
     '\n',
     '------ATOMS------   ATOM TYPES                                 WELL\n',
     '  I        J          I    J    DISTANCE   ENERGY     R*      DEPTH\n',
     '--------------------------------------------------------------------\n',
     'O  #1    C  #4        6   63      3.752    -0.056    3.936    0.063\n',
     'O  #1    C  #5        6   64      4.711    -0.033    3.936    0.063\n',
     'O  #1    C  #6        6    1      4.977    -0.020    3.771    0.068\n',
     'O  #1    N  #7        6   81      5.902    -0.006    3.621    0.074\n']

    'E L E C T R O S T A T I C\n',
     '\n',
     '------ATOMS------   ATOM TYPES                                 WELL\n',
     '  I        J          I    J    DISTANCE   ENERGY     R*      DEPTH\n',
     '--------------------------------------------------------------------\n',
     'O  #1    C  #4        6   63      3.752    -0.056    3.936    0.063\n',
     'O  #1    C  #5        6   64      4.711    -0.033    3.936    0.063\n',
     'O  #1    C  #6        6    1      4.977    -0.020    3.771    0.068\n',
     'O  #1    N  #7        6   81      5.902    -0.006    3.621    0.074\n']

    Returns
    {1: ..., 2: ..., 3:..}
    """
    smile = mol.to_smiles()

    if not isinstance(mol, Mol):
        try:
            mol = mol.to_rdkit() # try converting to rdkit
        except:
            raise Exception(f"Object of type {type(mol)} does not support `rdkit`")

    rdkit_cout = _get_rdkit_cout(smile)
    rdkit_cout = rdkit_cout.split("\n")


    parse = _parse_rdkit_output(rdkit_cout)
    expected_tot_en = float(rdkit_cout[-2]) # last output is always total energy
    
    energy_index = -1
    tot_en = 0
    
    atoms = defaultdict(float)
    energies = {k: [] for k in parse.keys()}
    for k, rows in parse.items():
        for r in rows:
            if COL_KEYWORD in r:
                energy_index = r.split().index(COL_KEYWORD) + 2
      
            elif energy_index > 0 and "#" in r:
                energy = float(r.split()[energy_index])
                energies[k].append(energy)
                atom_indexes = [int(x.replace("#", "")) for x in  re.findall("#\d+", r)]
                
                # energy contribution only for first atom?
                atoms[atom_indexes[0]-1] += energy
                tot_en += energy
    
    if not math.isclose(tot_en, expected_tot_en, rel_tol=ERR_TOLLERANCE) or not math.isclose(sum(atoms.values()), expected_tot_en, rel_tol=ERR_TOLLERANCE):
        raise Exception(f"Total energy {en} does not correspond to expected {expected_tot_en}")
    return atoms

def get_energy_and_derivative_from_conformers(m, poses, angle_term=True, bond_term=True, oop_term=True,
                                              bend_term=True, torsion_term=True, vdw_term=True, ele_term=True):
    m2 = Chem.AddHs(m)
    AllChem.EmbedMolecule(m2)
    AllChem.MMFFOptimizeMolecule(m2)
    pr = ff.MMFFGetMoleculeProperties(m2)

    pr.SetMMFFAngleTerm(angle_term)
    pr.SetMMFFBondTerm(bond_term)
    pr.SetMMFFOopTerm(oop_term)
    pr.SetMMFFStretchBendTerm(bend_term)
    pr.SetMMFFTorsionTerm(torsion_term)
    pr.SetMMFFVdWTerm(vdw_term)
    pr.SetMMFFVdWTerm(ele_term)
    
    mmff = ff.MMFFGetMoleculeForceField(m2,pr)
    
    out = list(map(lambda pos: get_energy_and_derivative_from_conformer(mmff, pos), poses))
    return [x[0] for x in out], [x[1] for x in out]


def get_energy_and_derivative_from_conformer(mmff, pos):
    energy = mmff.CalcEnergy(pos)
    grad = mmff.CalcGrad(pos)
    return energy, grad
