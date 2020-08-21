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
