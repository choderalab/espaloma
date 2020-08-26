# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE CLASSES
# =============================================================================
def esol(*args, **kwargs):
    import os

    import pandas as pd

    path = os.path.dirname(esp.__file__) + "/data/esol.csv"
    df = pd.read_csv(path)
    smiles = df.iloc[:, -1]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)


def alkethoh(*args, **kwargs):
    import os

    import pandas as pd

    path = os.path.dirname(esp.__file__) + "/data/alkethoh.smi"
    df = pd.read_csv(path)
    smiles = df.iloc[:, 0]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)


def zinc(first=-1, *args, **kwargs):
    import tarfile
    from os.path import exists
    from openforcefield.topology import Molecule
    from rdkit import Chem

    fname = 'parm_at_Frosst.tgz'
    url = 'http://www.ccl.net/cca/data/parm_at_Frosst/parm_at_Frosst.tgz'

    if not exists(fname):
        import urllib.request
        urllib.request.urlretrieve(url, fname)

    archive = tarfile.open(fname)
    zinc_file = archive.extractfile('parm_at_Frosst/zinc.sdf')
    _mols = Chem.ForwardSDMolSupplier(zinc_file, removeHs=False)

    count = 0
    gs = []

    for mol in _mols:
        try:
            gs.append(
                esp.Graph(
                    Molecule.from_rdkit(mol, allow_undefined_stereo=True)
                )
            )

            count += 1

        except:
            pass

        if first != -1 and count >= first:
            break

    return esp.data.dataset.GraphDataset(gs, *args, **kwargs)

def qcarchive(
        collection_type="OptimizationDataset",
        name="OpenFF Full Optimization Benchmark 1",
        first=-1,
        *args, **kwargs
    ):
    from espaloma.data import qcarchive_utils
    client = qcarchive_utils.get_client()
    collection, record_names = qcarchive_utils.get_collection(client)
    if first != -1:
        record_names = record_names[:first]
    graphs = [
        qcarchive_utils.get_graph(collection, record_name)
        for record_name in record_names
    ]

    graphs = [graph for graph in graphs if graph is not None]

    return esp.data.dataset.GraphDataset(graphs, *args, **kwargs)
