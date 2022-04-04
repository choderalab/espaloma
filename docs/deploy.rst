Deploy espaloma 0.2.0 force field to parametrize your MM system
===============================================================
Pretrained espaloma force field could be deployed on arbitrary small molecule
systems in a few lines::

    # imports
    import os
    import torch
    import espaloma as esp

    # grab pretrained model
    if not os.path.exists("espaloma_model.pt"):
        os.system("wget http://data.wangyq.net/espaloma_model.pt")

    # define or load a molecule of interest via the Open Force Field toolkit
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    # create an Espaloma Graph object to represent the molecule of interest
    molecule_graph = esp.Graph(molecule)

    # apply a trained espaloma model to assign parameters
    espaloma_model = torch.load("espaloma_model.pt")
    espaloma_model(molecule_graph.heterograph)

    # create an OpenMM System for the specified molecule
    openmm_system = esp.graphs.deploy.openmm_system_from_graph(molecule_graph)
