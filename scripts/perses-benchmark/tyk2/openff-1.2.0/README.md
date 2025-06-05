# Perses benchmarks

This subdirectory exposes a CLI tool for running automated benchmarks from
[OpenFF's protein ligand benchmark dataset](https://github.com/openforcefield/protein-ligand-benchmark) using perses.

## Running all edges

A script to run all transformations in an LSF batch scheduler is provided, but will likely need to be modified for your batch queue system:
```bash
bsub < LSF-job-template.sh
```

## Running single edges

Assuming you have a clone of the perses code repository and you are standing in the `benchmarks` subdirectory
(where this file lives). Then the benchmarks can be run using the following command syntax:
```bash
python run_benchmarks.py --target [protein-name] --edge [edge-index]
```

For example, for running the seventh edge (zero-based, according to [plbenchmark data](https://github.com/openforcefield/protein-ligand-benchmark) )
for `tyk2` protein, you would run:
```bash
# Set up and run edge 6
python run_benchmarks.py --target tyk2 --edge 6
```
Should the calculation for an edge fail, you can simply re-run the same command-line and the calculation will resume:
```bash
# Resume failed edge 6
python run_benchmarks.py --target tyk2 --edge 6
```
For more information on how to use the tool, you can run `python run_benchmarks.py -h`.

## Analyzing benchmarks

To analyze the simulations a script called `benchmark_analysis.py` is used as follows:
```bash
python benchmark_analysis.py --target [protein-name]
```

For example, for tyk2 results:
```bash
python benchmark_analysis.py --target tyk2
```
This will generate an output CSV file for [`arsenic`](https://github.com/openforcefield/arsenic) and corresponding absolute and relative free energy plots as PNG files produced according to best practices.)

For more information on how to use the cli analysis tool use `python benchmark_analysis.py -h`.
