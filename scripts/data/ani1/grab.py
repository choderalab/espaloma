import torch
import h5py
import espaloma as esp
import numpy as np

def run(path, u_thres=0.1):
    _ds = h5py.File(path)
    for name in _ds.keys():
        ds = _ds[name]
        _idx = 0
        for key in ds:
            print(_idx)
            mol = ds[key]
            smiles = mol["smiles"]
            smiles = np.array(smiles).tolist()
            smiles = [x.decode("UTF-8") for x in smiles]
            smiles = "".join(smiles)

            xs = np.array(mol["coordinates"])
            us = np.array(mol["energies"])
            species = list(mol["species"])
            species = [x.decode("UTF-8") for x in species]
          
            
            idxs = list(range(len(xs)))
            idx_ref = us.argmin()
            ok_idxs = [idx for idx in idxs if us[idx] <= us[idx_ref] + u_thres]

            from espaloma.data.utils import infer_mol_from_coordinates
            g = infer_mol_from_coordinates(xs[idx_ref], species, smiles_ref=smiles)
            g.nodes['n1'].data['xyz'] = torch.tensor(xs[ok_idxs, :, :]).transpose(1, 0)
            g.nodes['g'].data['u_ref'] = torch.tensor(us[None, ok_idxs])
            g.save("ani1/%s%s" % (name, _idx))
            _idx += 1


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
