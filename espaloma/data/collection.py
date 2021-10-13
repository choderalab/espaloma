# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp


# =============================================================================
# MODULE CLASSES
# =============================================================================
def esol(*args, **kwargs):
    """ ESOL collection.

    ..[1] ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure
        John S. Delaney
        Journal of Chemical Information and Computer Sciences
        2004 44 (3), 1000-1005
        DOI: 10.1021/ci034243x
    """
    import os

    import pandas as pd

    path = os.path.dirname(esp.__file__) + "/data/esol.csv"
    df = pd.read_csv(path)
    smiles = df.iloc[:, -1]
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)


def alkethoh(*args, **kwargs):
    """ AlkEthOH collection.

    ..[1] Open Force Field Consortium: Escaping atom types using direct chemical
    perception with SMIRNOFF v0.1
    David L. Mobley, Caitlin C. Bannan, Andrea Rizzi, Christopher I. Bayly,
    John D. Chodera, Victoria T. Lim, Nathan M. Lim, Kyle A. Beauchamp,
    Michael R. Shirts, Michael K. Gilson, Peter K. Eastman
    bioRxiv 286542; doi: https://doi.org/10.1101/286542

    """
    import os

    import pandas as pd

    df = pd.concat(
        [
            pd.read_csv(
                "https://raw.githubusercontent.com/openff.toolkit/"
                "open-forcefield-data/master/Model-Systems/AlkEthOH_distrib/"
                "AlkEthOH_rings.smi",
                header=None,
            ),
            pd.read_csv(
                "https://raw.githubusercontent.com/openff.toolkit/"
                "open-forcefield-data/master/Model-Systems/AlkEthOH_distrib/"
                "AlkEthOH_chain.smi",
                header=None,
            ),
        ],
        axis=0,
    )

    smiles = df.iloc[:, 0].values
    return esp.data.dataset.GraphDataset(smiles, *args, **kwargs)


def zinc(first=-1, *args, **kwargs):
    """ ZINC collection.

    ..[1] Irwin, John J, and Brian K Shoichet.
    “ZINC
    --a free database of commercially available compounds for virtual screening.”
    Journal of chemical information and modeling
    vol. 45,1 (2005): 177-82. doi:10.1021/ci049714+
    """
    import tarfile
    from os.path import exists
    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    fname = "parm_at_Frosst.tgz"
    url = "http://www.ccl.net/cca/data/parm_at_Frosst/parm_at_Frosst.tgz"

    if not exists(fname):
        import urllib.request

        urllib.request.urlretrieve(url, fname)

    archive = tarfile.open(fname)
    zinc_file = archive.extractfile("parm_at_Frosst/zinc.sdf")
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
        esp.data.md17_utils.get_molecule(name, *args, **kwargs)
        for name in [
            "benzene",
            "uracil",
            "naphthalene",
            "aspirin",
            "salicylic",
            "malonaldehyde",
            "ethanol",
            "toluene",
            "paracetamol",
            "azobenzene",
        ]
    ]


def md17_new(*args, **kwargs):
    return [
        esp.data.md17_utils.get_molecule(name, *args, **kwargs).heterograph
        for name in [
            "paracetamol",
            "azobenzene",
            "benzene",
            "ethanol",
        ]
    ]


class qca(object):
    pass


df_names = [
    "Bayer",
    "Coverage",
    "eMolecules",
    "Pfizer",
    "Roche",
    "Benchmark",
    "fda",
]


def _get_ds(cls, df_name):
    import os
    import pandas as pd

    path = os.path.dirname(esp.__file__) + "/../data/qca/%s.h5" % df_name
    df = pd.read_hdf(path)
    ds = esp.data.qcarchive_utils.h5_to_dataset(df)
    return ds


from functools import partial

for df_name in df_names:
    setattr(
        qca,
        df_name.lower(),
        classmethod(partial(_get_ds, df_name=df_name)),
    )
