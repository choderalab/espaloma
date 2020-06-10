import os
import urllib

from pkg_resources import resource_filename

alkethoh_url = 'https://raw.githubusercontent.com/openforcefield/open-forcefield-data/e07bde16c34a3fa1d73ab72e2b8aeab7cd6524df/Model-Systems/AlkEthOH_distrib/AlkEthOH_rings.smi'
path_to_smiles = resource_filename('espaloma.data.alkethoh', 'AlkEthOH_rings.smi')
path_to_offmols = resource_filename('espaloma.data.alkethoh', 'AlkEthOH_rings_offmols.pkl')
path_to_npz = resource_filename('espaloma.data.alkethoh', 'AlkEthOH_rings.npz')


def download_alkethoh():
    if not os.path.exists(path_to_smiles):
        with urllib.request.urlopen(alkethoh_url) as response:
            smi = response.read()
        with open(path_to_smiles, 'wb') as f:
            f.write(smi)

from pickle import load
with open(path_to_offmols, 'rb') as f:
    offmols = load(f)