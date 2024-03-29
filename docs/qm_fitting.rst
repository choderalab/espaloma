Quantum mechanics (QM) fitting experiment.
==========================================

**Open in Google Colab:**
http://data.wangyq.net/esp_notesbooks/qm_fitting.ipynb

This notebook recovers the QM fitting experiment in
https://arxiv.org/abs/2010.01196

|image1| **Table 2:** Espaloma can directly fit quantum chemical
energies to produce a new molecular mechanics force fields with better
accuracy than traditional force fields based on atom typing or direct
chemical perception. Espaloma was fit to quantum chemical potential
energies for conformations generated by optimization trajectories from
multiple conformers in various datasets from QCArchive.All datasets were
partitioned by molecules 80:10:10 into train:validate:test sets. We
report the RMSE on training and test sets, as well as the performance of
legacy force fields on the test set. All statistics are computed with
predicted and reference energies centered to have zero mean for each
molecule to focus on errors in relative conformational energetics,
rather than on errors in predicting the heats of formation of chemical
species (which the MM functional form used here is incapable of). The
95% confidence intervals annotated are calculated by via bootstrapping
molecules with replacement using 1000 replicates. \*: Six cyclic
peptides that cannot be parametrized using OpenForceField toolkit
engine~:raw-latex:`\cite{openff-toolkit-0.10.0}` and is not included.

Since Espaloma can derive a force field solely by fitting to energies
(and optionally gradients), we repeat the end-to-end fitting experiment
(See notebook
http://data.wangyq.net/esp_notebooks/phalkethoh_mm_small.ipynb) directly
using a quantum chemical (QM) datasets used to build and evaluate MM
force fields. We assessed the ability of Espaloma to learn several
distinct quantum chemical datasets generated by the Open Force Field
Initiativeand deposited in the MolSSI QCArchive: - **PhAlkEthOH** is a
collection of compounds containing only the elements carbon, hydrogen,
and oxygen in compounds containing phenyl rings, alkanes, ketones, and
alcohols. Limited in elemental and chemical diversity, this dataset is
chosen as a proof-of-concept to demonstrate the capability of Espaloma
to fit and generalize quantum chemical energies when training data is
sufficient to exhaustively cover the breadth of chemical environments. -
**OpenFF Gen2 Optimization** consists of druglike molecules used in the
parametrization of the Open Force Field 1.2.0 (“Parsley”) small molecule
force field. This set was constructed by the Open Force Field Consortium
from challenging molecule structures provided by Pfizer, Bayer, and
Roche, along with diverse molecules selected from eMolecules to achieve
useful coverage of chemical space. - **VEHICLe**, or *virtual
exploratory heterocyclic library*, is a set of heteroaromatic ring
systems of interest to drug discovery. The atoms in the molecules in
this dataset have interesting chemical environments in heteroarmatic
rings that present a challenge to traditional atom typing schemes, which
cannot easily accomodate the nuanced distinctions in chemical
environments that lead to perturbations in heterocycle structure.We use
this dataset to illustrate that Espaloma performs in situations
challenging to traditional force fields. - **PepConf** contains a
variety of short peptides, including capped, cyclic, and
disulfide-bonded peptides.This dataset—regenerated using the Open Force
Field QCSubmit tool—explores the applicability of Espaloma to
biopolymers, such as proteins.

Since nonbonded terms are generally optimized to fit other
condensed-phase properties, we focused here on optimizing only the
valence parameters (bond, angle, and proper and improper torsion) to fit
these gas-phase quantum chemical datasets, fixing the non-bonded
energies using a legacy force field. Because we are learning an MM force
field that is incapable of reproducing quantum chemical heats of
formation reflected as an additive offset in the quantum chemical energy
targets, in both training and test sets, snapshot energies for each
molecule are shifted to have zero mean. All datasets are randomly
shuffled and split (by molecules) into training (80%), validation (10%),
and test (10%) sets.

.. |image1| image:: https://pbs.twimg.com/media/FBL1Gb0WEAYkUhM?format=png&name=4096x4096

Installation and imports
------------------------

.. code:: python

    # install conda
    ! pip install -q condacolab
    import condacolab
    condacolab.install()

.. code:: python

    %%capture
    ! mamba install --yes --strict-channel-priority --channel jaimergp/label/unsupported-cudatoolkit-shim --channel omnia --channel omnia/label/cuda100 --channel dglteam --channel numpy openmm openmmtools openmmforcefields rdkit openff-toolkit dgl-cuda10.0 qcportal

.. code:: python

    ! git clone https://github.com/choderalab/espaloma.git

.. code:: python

    import torch
    import sys
    sys.path.append("/content/espaloma")
    import espaloma as esp

Load dataset
------------

Choose a dataset from ``["gen2", "pepconf", "vehicle", "phalkethoh"]``.

.. code:: python

    dataset_name = "gen2"
    # dataset_name = "pepconf"
    # dataset_name = "vehicle"
    # dataset_name = "phalkethoh"

.. code:: python

    %%capture
    ! wget "data.wangyq.net/esp_dataset/"$dataset_name".zip"
    ! unzip $dataset_name".zip"

.. code:: python

    ds = esp.data.dataset.GraphDataset.load(dataset_name)
    ds.shuffle(seed=2666)
    ds_tr, ds_vl, ds_te = ds.split([8, 1, 1])

Define model
------------

Define Espaloma stage I: graph -> atom latent representation

.. code:: python

    representation = esp.nn.Sequential(
        layer=esp.nn.layers.dgl_legacy.gn("SAGEConv"), # use SAGEConv implementation in DGL
        config=[128, "relu", 128, "relu", 128, "relu"], # 3 layers, 128 units, ReLU activation
    )

Define Espaloma stage II and III: atom latent representation -> bond,
angle, and torsion representation and parameters. And compose all three
Espaloma stages into an end-to-end model.

.. code:: python

    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=128, config=[128, "relu", 128, "relu", 128, "relu"],
        out_features={              # define modular MM parameters Espaloma will assign
            1: {"e": 1, "s": 1}, # atom hardness and electronegativity
            2: {"log_coefficients": 2}, # bond linear combination, enforce positive
            3: {"log_coefficients": 2}, # angle linear combination, enforce positive
            4: {"k": 6}, # torsion barrier heights (can be positive or negative)
        },
    )
    
    espaloma_model = torch.nn.Sequential(
                     representation, readout, esp.nn.readout.janossy.ExpCoefficients(),
                     esp.mm.geometry.GeometryInGraph(), 
                     esp.mm.energy.EnergyInGraph(),
    )


.. code:: python

    if torch.cuda.is_available():
        espaloma_model = espaloma_model.cuda()

Loss function is specified as the MSE between predicted and reference
energy.

.. code:: python

    loss_fn = esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(), # use mean-squared error loss
            between=['u', "u_ref"],         # between predicted and QM energies
            level="g", # compare on graph level
    )

Define optimizer
----------------

.. code:: python

    optimizer = torch.optim.Adam(espaloma_model.parameters(), 1e-4)

Train it!
---------

.. code:: python

    for idx_epoch in range(10000):
        for g in ds_tr:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                g.heterograph = g.heterograph.to("cuda:0")
            g = espaloma_model(g.heterograph)
            loss = loss_fn(g)
            loss.backward()
            optimizer.step()
        torch.save(espaloma_model.state_dict(), "%s.th" % idx_epoch)

Inspect
-------

.. code:: python

    inspect_metric = esp.metrics.center(torch.nn.L1Loss()) # use mean-squared error loss

.. code:: python

    loss_tr = []
    loss_vl = []

.. code:: python

    with torch.no_grad():
        for idx_epoch in range(10000):
            espaloma_model.load_state_dict(
                torch.load("%s.th" % idx_epoch)
            )
    
            # training set performance
            u = []
            u_ref = []
            for g in ds_tr:
                if torch.cuda.is_available():
                    g.heterograph = g.heterograph.to("cuda:0")
                espaloma_model(g.heterograph)
                u.append(g.nodes['g'].data['u'])
                u_ref.append(g.nodes['g'])
            u = torch.cat(u, dim=0)
            u_ref = torch.cat(u_ref, dim=0)
            loss_tr.append(inspect_metric(u, u_ref))
    
    
            # validation set performance
            u = []
            u_ref = []
            for g in ds_vl:
                if torch.cuda.is_available():
                    g.heterograph = g.heterograph.to("cuda:0")
                espaloma_model(g.heterograph)
                u.append(g.nodes['g'].data['u'])
                u_ref.append(g.nodes['g'])
            u = torch.cat(u, dim=0)
            u_ref = torch.cat(u_ref, dim=0)
            loss_vl.append(inspect_metric(u, u_ref))


.. code:: python

    import numpy as np
    loss_tr = np.array(loss_tr) * 627.5
    loss_vl = np.array(loss_vl) * 627.5

.. code:: python

    from matplotlib import pyplot as plt 
    plt.plot(loss_tr, label="train")
    plt.plot(loss_vl, label="valid")
    plt.yscale("log")
    plt.legend()
