# hgfp
Hypergraph Functional Potential

## Contents
* `graph.py` contains functions for extracting `dgl` graphs from `rdkit` or `openeye` molecules, and for extracting similar per-atom features using `rdkit` or `openeye`.
* `heterograph.py` contains a function to convert a `dgl` molecular graph (where nodes are atoms and edges are bonds) into a graph mimicking the MM model's factor graph (where some nodes correspond to interactions, some nodes correspond to atoms, and edges indicate which atoms participate in which interactions).
* `hierarchical_graph.py` contains a function to convert an `openeye` molecule into a more elaborate `dgl` heterograph (TODO: understand a bit more what this is doing).
* `supervised_param_train.py` contains a function to parse arguments from the command-line function to fit a model to "`k`" and "`eq`" parameters for `bond`, `angle`, `torsion` interactions, and write the results and a summary to disk.

### app
* `supervised_param_train.py` is similar to `supervised_param_train.py` at top-level.
* `supervised_train.py` is similar to `supervised_param_train.py`, but fitting to potential energies rather than parameters.

### data
* `mm_energy.py` contains a function `u` that calls `mm.energy_in_heterograph` to compute the sum of bond, angle, torsion, and Lennard-Jones terms.
* `utils.py` contains 2 classes `BatchedDataset` and `BatchedParamGraph`, and 4 functions `split`, `get_norm_dict`, `get_norm_fn`, and `get_norm_fn_log_normal`.  `BatchedParamGraph` supports iteration over `dgl` batches of a given size. `split` returns a tuple of train/validation/test lists/iterators (possibly overlapping?). `get_norm_dict` returns a nested dictionary of means and standard deviations for each parameter in each interaction term. `get_norm_fn` returns a pair of functions that modify a `dgl` graph in place by adding/multiplying or subtracting/dividing by mean and stddev attributes. `get_norm_fn_log_normal` does something similar, but possibly assuming the parameters are log-normally distributed.

each of the following folders contains a file `df.py` and a file `param.py` (exceptions: qc_archive/ and gm9_mm/ do not contain `param.py`):
* ani/ contains utilities for reading snapshots and energies from an ANI dataset (which one?) and inferring molecular topology from coordinates.
* parm_at_Frosst/ contains utilities for producing `dgl` heterographs from the `parm@Frosst` dataset.
* qc_archive/ contains utilities for sampling records from "OptimizationDataset" and "OpenFF Full Optimization Benchmark 1" QCArchive datasets.
* qm9/ contains utilities for loading QM9 snapshots and energies.
* qm9_mm/ loads the same molecules as qm9/ but evaluates their energies using a MM forcefield (interpreting the forcefield parameters using the current package, rather than an MD engine).

### mm
* `idxs.py` contains a function to compute indices of which atoms participate in bonds, angles, torsions, 1-4 exceptions, and non-bonded interactions.
* `geometry.py` contains functions to compute distances, angles, and torsions given coordinates and interaction indices.
* `geometry_in_heterograph.py` contains functions to compute distances, angles, and torsions from heterographs or their nodes. Note use of `angle_vl`, `torsion_vl`, compared with `angle`, `torsion`.
* `energy.py` contains pytorch functions for harmonic bonds and angles, torsions with periodicity 1, and the Lennard-Jones potential.
* `energy_ii.py` contains pytorch functions for harmonic bonds, angles, and angle-bond couplings used in "class-II" forcefields, as well as Coulomb and Lennard-Jones potentials.
* `energy_in_heterograph.py` contains a function `u` that accepts a `dgl` heterograph that has geometry terms computed on each node corresponding to an interaction, and sums up contributions from harmonic bond, harmonic angle, torsion, and Lennard-Jones terms.
* `energy_in_heterograph_ii.py` contains a function that performs message-passing on a graph produced by `hierarchical_graph.py` (?) to compute energy.

### models
* `gcn.py` contains 3 classes, `NodeThenFullyConnect`, `GCN`, and `Net`. `Net` is controlled by checking whether elements of the `config` argument are ints or strings.
* `gcn_with_combine_readout.py` contains 3 classes, `GN`, `ParamReadout`, and `Net`. `ParamReadout` writes interaction parameters onto nodes representing parameters, and `Net` returns the result of performing a potential energy calculation using these parameters.
* `hierarchical_message_passing.py` contains a class `HMP` that passes messages "up" and "down" on a `dgl` graph and returns the updated `dgl` graph
* `walk_recurrent.py` contains classes `WRGN`, `ParamReadout`, and `Net`, similar to `gcn_with_combine_readout.py`, but using a different model for reading out interaction parameters.

### test
* `test_hierarchical_graph.py` prints the result of calling hierarchical graph constructor.
* `test_hypergraph.py` prints the result of calling heterograph constructor.
