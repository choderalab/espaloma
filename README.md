espaloma
==============================
[//]: # (Badges)
[![CI](https://github.com/choderalab/espaloma/actions/workflows/CI.yaml/badge.svg?branch=master)](https://github.com/choderalab/espaloma/actions/workflows/CI.yaml)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/choderalab/espaloma.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/choderalab/espaloma/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/choderalab/espaloma.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/choderalab/espaloma/context:python)

Extensible Surrogate Potential of Optimized by Message-passing Algorithms

![abstract](docs/_static/espaloma_abstract_v2-2.png)

# Abstract
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




# License

This software is licensed under [MIT license](https://opensource.org/licenses/MIT).

# Copyright

Copyright (c) 2020, Chodera Lab at Memorial Sloan Kettering Cancer Center and Authors:
Authors:
- Yuanqing Wang
- Josh Fass
- John D. Chodera
