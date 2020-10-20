import qcportal

dataset_type = 'OptimizationDataset'

optimization_datasets = {
    "Roche": "OpenFF Gen 2 Opt Set 1 Roche",
    "Coverage": "OpenFF Gen 2 Opt Set 2 Coverage",
    "Pfizer": "OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy",
    "eMolecules": "OpenFF Gen 2 Opt Set 4 eMolecules Discrepancy",
    "Bayer": "OpenFF Gen 2 Opt Set 5 Bayer",
}

batch_size = 100


def batch_ify(records: list, batch_size: int):
    batches = [records[i * batch_size:(i + 1) * batch_size] for i in range(1 + (len(records) // batch_size))]
    assert (sum(map(len, batches)) == len(records))
    return batches


if __name__ == '__main__':
    client = qcportal.FractalClient()

    records_by_dataset = dict()
    for (short_name, long_name) in optimization_datasets.items():
        ds = client.get_collection(dataset_type, long_name)
        specifications = ds.list_specifications(description=False)

        records = list(ds.data.records)
        records_by_dataset[long_name] = records

        print(short_name, len(specifications), len(records))

    batches = {short: batch_ify(records_by_dataset[long], batch_size) for (short, long) in
               optimization_datasets.items()}

    print(f'# of batches of size {batch_size}')
    print([(short_name, len(batch)) for (short_name, batch) in batches.items()])

    for short_name in batches:
        for i in range(len(batches[short_name])):
            with open(f'batches/{short_name}/{i}.txt', 'w') as f:
                f.writelines('\n'.join(batches[short_name][i]))
