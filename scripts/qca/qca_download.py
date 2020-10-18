import qcportal as ptl

import espaloma as esp


def get_collection() -> ptl.collections.OptimizationDataset:
    """fetches "OpenFF Full Optimization Benchmark 1"""

    client = ptl.FractalClient()
    collection = client.get_collection(
        "OptimizationDataset",
        "OpenFF Full Optimization Benchmark 1"
    )

    return collection


def get_graph(collection, idx) -> None:
    """creates an esp.graph from the idx record in collection and saves to data/{idx}.th"""

    print(idx, flush=True)
    record_names = list(collection.data.records)
    record_name = record_names[idx]

    g = esp.data.qcarchive_utils.get_graph(collection, record_name)

    g.save('data/%s.th' % idx)

    print(idx, "done")


def run(idx, batch_size) -> None:
    """saves to data/{idx}.th for idx in range(batch_size * idx, batch_size * (idx + 1))"""

    idx = int(idx)
    batch_size = int(batch_size)

    collection = get_collection()
    before = batch_size * idx
    for _idx in range(before, before + batch_size):
        import os
        if not os.path.exists('data/%s.th' % _idx):
            get_graph(collection, _idx)


if __name__ == '__main__':
    import sys

    run(sys.argv[1], sys.argv[2])
