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


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mean_and_std():
    return 0.0, 1.0

def unbatched(num=-1, hetero=False):
    """ Put qm9 molecules in a dataset.
    """

    client = ptl.FractalClient()
    from openforcefield.topology import Molecule
    from openforcefield.topology import Topology
    from openforcefield.typing.engines.smirnoff import ForceField
    FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')
    import cmiles
    from simtk import openmm
    import random

    ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")

    # initialize graph list to be empty
    def _iter():
        for record_name in random.sample(list(ds_qc.data.records), num):
            try:

                print(record_name, flush=True)
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

                            if hetero is True:
                                g = hgfp.heterograph.from_graph(g)


                            yield(g, u)

            except:
                pass


    return _iter

def batched(
        num=-1,
        n_batches_in_buffer=12,
        batch_size=32,
        cache=True,
        hetero=False):

    return hgfp.data.utils.BatchedDataset(
        unbatched(num=num, hetero=hetero),
        n_batches_in_buffer=n_batches_in_buffer,
        batch_size=batch_size,
        cache=cache,
        hetero=hetero)
