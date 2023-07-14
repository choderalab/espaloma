espaloma: **E**xtensible **S**urrogate **P**otenti**al** **O**ptimized by **M**essage-passing **A**lgorithms 🍹
==============================
[//]: # (Badges)
[![CI](https://github.com/choderalab/espaloma/actions/workflows/CI.yaml/badge.svg?branch=main)](https://github.com/choderalab/espaloma/actions/workflows/CI.yaml)
[![Documentation Status](https://readthedocs.org/projects/espaloma/badge/?version=latest)](https://espaloma.readthedocs.io/en/latest/?badge=latest)

Source code for [Wang Y, Fass J, and Chodera JD "End-to-End Differentiable Construction of Molecular Mechanics Force Fields."](https://arxiv.org/abs/2010.01196)

![abstract](docs/_static/espaloma_abstract_v2-2.png)

#
Documentation: https://docs.espaloma.org

# Paper Abstract
Molecular mechanics (MM) potentials have long been a workhorse of computational chemistry.
Leveraging accuracy and speed, these functional forms find use in a wide variety of applications in biomolecular modeling and drug discovery, from rapid virtual screening to detailed free energy calculations.
Traditionally, MM potentials have relied on human-curated, inflexible, and poorly extensible discrete chemical perception rules _atom types_ for applying parameters to small molecules or biopolymers, making it difficult to optimize both types and parameters to fit quantum chemical or physical property data.
Here, we propose an alternative approach that uses _graph neural networks_ to perceive chemical environments, producing continuous atom embeddings from which valence and nonbonded parameters can be predicted using invariance-preserving layers.
Since all stages are built from smooth neural functions, the entire process---spanning chemical perception to parameter assignment---is modular and end-to-end differentiable with respect to model parameters, allowing new force fields to be easily constructed, extended, and applied to arbitrary molecules.
We show that this approach is not only sufficiently expressive to reproduce legacy atom types, but that it can learn and extend existing molecular mechanics force fields, construct entirely new force fields applicable to both biopolymers and small molecules from quantum chemical calculations, and even learn to accurately predict free energies from experimental observables.

# Manifest

* `espaloma/` core code for graph-parametrized potential energy functions.
    * `graphs/` data objects that contain various level of information we need.
        * `graph.py` base modules for graphs.
        * `molecule_graph.py` provide APIs to various molecular modelling toolkits.
        * `homogeneous_graph.py` simplest graph representation of a molecule.
        * `heterogeneous_graph.py` graph representation of a molecule that contains information regarding membership of lower-level nodes to higher-level nodes.
        * `parametrized_graph.py` graph representation of a molecule with all parameters needed for energy evaluation.
    * `nn/` neural network models that facilitates translation between graphs.
        * `dgl_legacy.py` API to dgl models for atom-level message passing.
    * `mm/` molecular mechanics functionalities for energy evaluation.
        * `i/` energy terms used in Class-I force field.
            * `bond.py` bond energy
            * `angle.py` angle energy
            * `torsion.py` torsion energy
            * `nonbonded.py` nonbonded energy
        * `ii/` energy terms used in Class-II force field.
            * `coupling.py` coupling terms
            * `polynomial.py` higher order polynomials.


# Example: Deploy espaloma 0.2.0 pretrained force field to arbitrary MM system

```python  
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
```

# Using espaloma to parameterize small molecules in relative free energy calculations

An example of using espaloma to parameterize small molecules in relative alchemical free energy calculations is provided in the `scripts/perses-benchmark/` directory.

# License

This software is licensed under [MIT license](https://opensource.org/licenses/MIT).

# Copyright

Copyright (c) 2020, Chodera Lab at Memorial Sloan Kettering Cancer Center and Authors:
Authors:
- [Yuanqing Wang](http://www.wangyq.net)
- Josh Fass
- John D. Chodera
