espaloma
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/espaloma.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/espaloma)

Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithms

Rather than:

molecule ---(atom typing schemes)---> atom-types ---(atom typing schemes)---> bond-, angle-, torsion-types ---(table lookup)---> force field parameters

we want to have

molecule ---(graph nets)---> atom-embedding ---(pooling)---> hypernode-embedding ---(feedforward neural networks)---> force field parameters


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

