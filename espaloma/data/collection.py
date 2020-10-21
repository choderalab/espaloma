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

def md17_old(*args, **kwargs):
    return [
        esp.data.md17_utils.get_molecule(
            name, *args, **kwargs
        ).heterograph for name in [
            # 'benzene',
            'uracil',
            'naphthalene',
            'aspirin', 'salicylic',
            'malonaldehyde',
            # 'ethanol',
            'toluene',
   'paracetamol', 'azobenzene'
        ]]

def md17_new(*args, **kwargs):
    return [
        esp.data.md17_utils.get_molecule(
            name, *args, **kwargs
        ).heterograph for name in [
            # 'paracetamol', 'azobenzene',
            'benzene', 'ethanol',
        ]]


class qca(object):
    pass

df_names = ['Bayer', 'Converage', 'eMolecules', 'Pfizer', 'Roche']
for df_name in df_names:
    def _get_ds(cls):
        import os
        import pandas as pd
        path = os.path.dirname(esp.__file__) + "/../data/qca/%s.h5" % df_name
        df = pd.read_hdf(path)
        ds = esp.data.qcarchive_utils.h5_to_dataset(df)
        return ds

    setattr(qca, df_name.lower(), classmethod(_get_ds))
