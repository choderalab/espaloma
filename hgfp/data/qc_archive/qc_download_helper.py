# =============================================================================
# IMPORTS
# =============================================================================
import rdkit
from rdkit import Chem
import pandas as pd
import dgl
import torch
import os
import hgfp
import qcportal as ptl
import numpy as np

client = ptl.FractalClient()
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')
import cmiles
from simtk import openmm
import random
import numpy as np
from multiprocessing import Pool
import itertools
from dgl import data
ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")


import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
# initialize graph list to be empty

records = list(ds_qc.data.records)

def f(record_name):
    try:
            print(record_name, flush=True)
            us = []
            gs = []
            r = ds_qc.get_record(record_name, specification='default')
            if r is not None:
                traj = r.get_trajectory()
                if traj is not None:
                    for snapshot in traj:
                        with timeout(seconds=5):
                            energy = snapshot.properties.scf_total_energy
                            mol = snapshot.get_molecule()

                            mol = cmiles.utils.load_molecule(mol.dict(encoding='json'),
                                toolkit='rdkit')

                            u = torch.squeeze(torch.Tensor([energy]))
                            g = hgfp.graph.from_rdkit_mol(mol)

                            us.append(u)
                            gs.append(g)

            return us, gs
    except:
        pass
            

if __name__ == '__main__':
    p = Pool(128)
    us_gs_array = p.map(f, records)
    us = torch.stack(list(itertools.chain.from_iterable([x[1] for x in us_gs_array if x is not None])))
    gs = list(itertools.chain.from_iterable([x[0] for x in us_gs_array if x is not None]))

    dgl.data.utils.save_graphs('qc_archive_1.bin', gs, {'u': us})
