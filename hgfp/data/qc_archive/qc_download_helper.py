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


ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")

# initialize graph list to be empty

records = list(ds_qc.data.records)

global ds
gs = []
us = []

for record_name in records:
    try:
            r = ds_qc.get_record(record_name, specification='default')

            if r is not None:
                traj = r.get_trajectory()
                if traj is not None:
                    for snapshot in traj:
                        energy = snapshot.properties.scf_total_energy
                        mol = snapshot.get_molecule()

                        mol = cmiles.utils.load_molecule(mol.dict(encoding='json'),
                            toolkit='rdkit')

                        u = torch.squeeze(torch.Tensor([energy]))
                        g = hgfp.graph.from_rdkit_mol(mol)

                        gs.append(g)
                        us.append(u)
    except:
        continue


dgl.data.utils.save_graphs('qc_archive.bin', gs, {'u': torch.stack(us)})
