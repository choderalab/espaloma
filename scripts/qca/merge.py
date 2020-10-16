import qcportal as ptl
import torch


def get_dict():
    client = ptl.FractalClient()

    collection = client.get_collection(
        "OptimizationDataset",
        "OpenFF Full Optimization Benchmark 1",
    )

    record_names = list(collection.data.records)

    mol_idx = -1 

    results = {}

    previous = ""

    for record_idx, record_name in enumerate(record_names):
        base_name = "".join(record_name.split("-")[:-1])
        if base_name != previous:
            mol_idx += 1
            results[mol_idx] = [record_idx]
            previous = base_name

        else:
            results[mol_idx].append(record_idx)

    return results

def run():
    _dict = get_dict()

    for record_idx, mol_idxs in _dict.items():
        import os
        paths = ['data/%s.th' % mol_idx for mol_idx in mol_idxs]
        paths = [path for path in paths if os.path.exists(path)]
        
        if len(paths) == 0:
            continue

        else:
            print(record_idx)

            import espaloma as esp

            gs = [
                esp.Graph().load(path)
                for path in paths
            ]

            g_ref = gs[0]


            g_ref.nodes['g'].data['u_ref'] = torch.cat(
                [
                    g.nodes['g'].data['u_ref'] for g in gs
                ],
                dim=1,
            )

            g_ref.nodes['n1'].data['xyz'] = torch.cat(
                [
                    g.nodes['n1'].data['xyz'] for g in gs
                ],
                dim=1,
            )

            g_ref.nodes['n1'].data['u_ref_prime'] = torch.cat(
                [
                    g.nodes['n1'].data['u_ref_prime'] for g in gs
                ],
                dim=1,
            )

            g_ref.save('merged_data/%s.th' % record_idx) 



if __name__ == '__main__':
    run()

        


