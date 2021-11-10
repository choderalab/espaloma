Installation
============

While we are working to put `espaloma` on conda-forge, please follow the instructions below to install dependencies and the `espaloma` package separately.

Install dependencies::

    conda install \
      --yes \
      --strict-channel-priority \
      --channel jaimergp/label/unsupported-cudatoolkit-shim \
      --channel omnia \
      --channel omnia/label/cuda100 \
      --channel dglteam \
      --channel numpy \
      openmm openmmtools openmmforcefields rdkit openff-toolkit dgl-cuda10.0 qcportal


Install the package::

    git clone https://github.com/choderalab/espaloma.git
    cd espaloma
    python setup.py install
