import glob
import os
import numpy as np
from chembl_webresource_client.new_client import new_client


def run(path):
    chembl_xyz_paths = glob.glob(os.path.join(path, "CHEMBL*"))
    chembl_ids_denali = {}
    for xyz_path in chembl_xyz_paths:
        _, tail = os.path.split(xyz_path)
        chembl_id = tail.split("_")[0]
        chembl_ids_denali[tail] = {"path": xyz_path,
                                    "chembl_id": chembl_id}

    mols = new_client.molecule
    chembl_ids = list(set([chembl_ids_denali[key]["chembl_id"] for key in chembl_ids_denali.keys()]))
    m1 = mols.filter(molecule_chembl_id__in=chembl_ids).only(['molecule_chembl_id', 'molecule_structures'])
    chemblid_smiles = {}
    while True:
        try:
            result = next(m1)
        except StopIteration:
            break
        except:
            continue
        chembl_id_result = result['molecule_chembl_id']
        smiles = result['molecule_structures']['canonical_smiles']
        chemblid_smiles[chembl_id_result] = smiles

    for key in chembl_ids_denali.keys():
        try:
            chembl_ids_denali[key]["canonical_smiles"] = chemblid_smiles[chembl_ids_denali[key]["chembl_id"]]
        except:
            pass
        xyz_path = chembl_ids_denali[key]["path"]
        xyz_files = glob.glob(os.path.join(xyz_path, "*.xyz"))
        coordinates = []
        sample_ids = []
        for xyz_file in xyz_files:
            _, tail = os.path.split(xyz_file)
            sample_ids.append(tail.rstrip(".xyz"))
            with open(xyz_file, "r") as f:
                next(f)
                line = next(f)
                multiplicity, charge = line.strip("\n").split()
                species = []
                coords = []
                while True:
                    try:
                        line = next(f).strip("\n").split()
                        species.append(line[0])
                        coords.append([float(line[1]), float(line[2]), float(line[3])])
                    except StopIteration:
                        break
                coordinates.append(np.array(coords))
        coordinates = np.array(coordinates)
        chembl_ids_denali[key]['species'] = species
        chembl_ids_denali[key]['coordinates'] = coordinates
        chembl_ids_denali[key]['charge'] = charge
        chembl_ids_denali[key]['multiplicity'] = multiplicity
        chembl_ids_denali[key]['sample_ids'] = sample_ids
        chembl_ids_denali[key]['energies'] = np.zeros((len(sample_ids)))

    with open("denali_labels.csv", "r") as f:
        next(f)
        while True:
            try:
                line = next(f).split(",")
                if line[3] in chembl_ids_denali.keys():
                    if line[1] in chembl_ids_denali[line[3]]['sample_ids']:
                        sample_idx = chembl_ids_denali[line[3]]['sample_ids'].index(line[1])
                        chembl_ids_denali[line[3]]['energies'][sample_idx] = float(line[9])
            except StopIteration:
                break

    import pickle

    with open("denali_dataset_dict.pkl", "wb") as f:
        pickle.dump(chembl_ids_denali, f)


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
