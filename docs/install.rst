Installation
============

mamba
-----

We recommend using `mamba <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-installation>`_ which is a drop-in replacement for ``conda`` and is much faster.

.. code-block:: bash

   $ mamba create --name espaloma -c conda-forge "espaloma=0.3.2"

Note: If you are using a Mac with an M1/M2 chip, you will need to install and run ``espaloma`` using `Rosetta <https://support.apple.com/en-us/HT211861>`_ by using the following commands:

.. code-block:: bash

   CONDA_SUBDIR=osx-64 mamba create --name espaloma -c conda-forge "espaloma=0.3.2"
   mamba activate espaloma
   mamba config --env --set subdir osx-64

This will ensure that any other packages installed in the ``espaloma`` environment will also use Rosetta.
