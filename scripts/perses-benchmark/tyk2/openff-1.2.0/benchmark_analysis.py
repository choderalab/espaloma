"""
Script to perform analysis of perses simulations executed using run_benchmarks.py script.

Intended to be used on systems from https://github.com/openforcefield/protein-ligand-benchmark
"""

import argparse
import glob
import itertools
import re
import warnings

import numpy as np
import urllib.request
import yaml

from openmmtools.constants import kB
from perses.analysis.load_simulations import Simulation

from simtk import unit

from openff.arsenic import plotting, wrangle

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


# Helper functions

def get_simdir_list(base_dir='.', is_reversed=False):
    """
    Get list of directories to extract simulation data.

    Attributes
    ----------
    base_dir: str, optional, default='.'
        Base directory where to search for simulations results. Defaults to current directory.
    is_reversed: bool, optional, default=False
        Whether to consider the reversed simulations or not. Meant for testing purposes.

    Returns
    -------
    dir_list: list
        List of directories paths for simulation results.
    """
    # Load all expected simulation from directories
    out_dirs = ['/'.join(filepath.split('/')[:-1]) for filepath in glob.glob(f'{base_dir}/out*/*complex.nc')]
    reg = re.compile(r'out_[0-9]+_[0-9]+_reversed')  # regular expression to deal with reversed directories
    if is_reversed:
        # Choose only reversed directories
        out_dirs = list(filter(reg.search, out_dirs))
    else:
        # Filter out reversed directories
        out_dirs = list(itertools.filterfalse(reg.search, out_dirs))
    return out_dirs


def get_simulations_data(simulation_dirs):
    """Generates a list of simulation data objects given the simulation directories paths."""
    simulations = []
    for out_dir in simulation_dirs:
        # Load complete or fully working simulations
        # TODO: Try getting better exceptions from openmmtools -- use non-generic exceptions
        try:
            simulation = Simulation(out_dir)
            simulations.append(simulation)
        except Exception:
            warnings.warn(f"Edge in {out_dir} could not be loaded. Check simulation output is complete.")
    return simulations


def to_arsenic_csv(experimental_data: dict, simulation_data: list, out_csv: str = 'out_benchmark.csv'):
    """
    Generates a csv file to be used with openff-arsenic. Energy units in kcal/mol.

    .. warning:: To be deprecated once arsenic object model is improved.

    Parameters
    ----------
        experimental_data: dict
            Python nested dictionary with experimental data in micromolar or nanomolar units.
            Example of entry:

                {'lig_ejm_31': {'measurement': {'comment': 'Table 4, entry 31',
                  'doi': '10.1016/j.ejmech.2013.03.070',
                  'error': -1,
                  'type': 'ki',
                  'unit': 'uM',
                  'value': 0.096},
                  'name': 'lig_ejm_31',
                  'smiles': '[H]c1c(c(c(c(c1[H])Cl)C(=O)N([H])c2c(c(nc(c2[H])N([H])C(=O)C([H])([H])[H])[H])[H])Cl)[H]'}

        simulation_data: list or iterable
            Python iterable object with perses Simulation objects as entries.
        out_csv: str
            Path to output csv file to be generated.
    """
    # Ligand information
    ligands_names = list(ligands_dict.keys())
    lig_id_to_name = dict(enumerate(ligands_names))
    kBT = kB * 300 * unit.kelvin  # useful when converting to kcal/mol
    # Write csv file
    with open(out_csv, 'w') as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for ligand_name, ligand_data in experimental_data.items():
            # TODO: Handle multiple measurement types
            unit_symbol = ligand_data['measurement']['unit']
            measurement_value = ligand_data['measurement']['value']
            measurement_error = ligand_data['measurement']['error']
            # Unit conversion
            # TODO: Let's persuade PLBenchmarks to use pint units
            unit_conversions = { 'M' : 1.0, 'mM' : 1e-3, 'uM' : 1e-6, 'nM' : 1e-9, 'pM' : 1e-12, 'fM' : 1e-15 }
            if unit_symbol not in unit_conversions:
                raise ValueError(f'Unknown units "{unit_symbol}"')
            value_to_molar= unit_conversions[unit_symbol]
            # Handle unknown errors
            # TODO: We should be able to ensure that all entries have more reasonable errors.
            if measurement_error == -1:
                # TODO: For now, we use a relative_error from the Tyk2 system 10.1016/j.ejmech.2013.03.070
                relative_error = 0.3
            else:
                relative_error = measurement_error / measurement_value
            # Convert to free eneriges
            expt_DG = kBT.value_in_unit(unit.kilocalorie_per_mole) * np.log(measurement_value * value_to_molar)
            expt_dDG = kBT.value_in_unit(unit.kilocalorie_per_mole) * relative_error
            csv_file.write(f"{ligand_name}, {expt_DG}, {expt_dDG}\n")

        # Calculated block
        # print header for block
        csv_file.write("# Calculated block\n")
        csv_file.write("# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n")
        # Loop through simulation, extract ligand1 and ligand2 indices, convert to names, create string with
        # ligand1, ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)
        # write string in csv file
        for simulation in simulation_data:
            out_dir = simulation.directory.split('/')[-1]
            # getting integer indices
            ligand1_id, ligand2_id = int(out_dir.split('_')[-1]), int(out_dir.split('_')[-2])  # CHECK ORDER!
            # getting names of ligands
            ligand1, ligand2 = lig_id_to_name[ligand1_id], lig_id_to_name[ligand2_id]
            # getting calc_DDG in kcal/mol
            calc_DDG = simulation.bindingdg.value_in_unit(unit.kilocalorie_per_mole)
            # getting calc_dDDG in kcal/mol
            calc_dDDG = simulation.bindingddg.value_in_unit(unit.kilocalorie_per_mole)
            csv_file.write(
                f"{ligand1}, {ligand2}, {calc_DDG}, {calc_dDDG}, 0.0\n")  # hardcoding additional error as 0.0


# Defining command line arguments
# fetching targets from github repo
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
targets_url = f"{base_repo_url}/raw/master/data/targets.yml"
with urllib.request.urlopen(targets_url) as response:
    targets_dict = yaml.safe_load(response.read())
# get the possible choices from targets yaml file
target_choices = targets_dict.keys()

arg_parser = argparse.ArgumentParser(description='CLI tool for running perses protein-ligand benchmarks analysis.')
arg_parser.add_argument(
    "--target",
    type=str,
    help="Target biomolecule, use openff's plbenchmark names.",
    choices=target_choices,
    required=True
)
arg_parser.add_argument(
    "--reversed",
    action='store_true',
    help="Analyze reversed edge simulations. Helpful for testing/consistency checks."
)
args = arg_parser.parse_args()
target = args.target

# Download experimental data
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
# TODO: Let's cache this data when we set up the initial simulations in case it changes in between setting up and running the calculations and analysis.
# TODO: Let's also be sure to use a specific release tag rather than 'master'
target_dir = targets_dict[target]['dir']
ligands_url = f"{base_repo_url}/raw/master/data/{target_dir}/00_data/ligands.yml"
with urllib.request.urlopen(ligands_url) as response:
    yaml_contents = response.read()
    print(yaml_contents)
    ligands_dict = yaml.safe_load(yaml_contents)

# DEBUG
print('')
print(yaml.dump(ligands_dict))

# Get paths for simulation output directories
out_dirs = get_simdir_list(is_reversed=args.reversed)

# Generate list with simulation objects
simulations = get_simulations_data(out_dirs)

# Generate csv file
csv_path = f'./{target}_arsenic.csv'
to_arsenic_csv(ligands_dict, simulations, out_csv=csv_path)


# TODO: Separate plotting in a different file
# Make plots and store
fe = wrangle.FEMap(csv_path)
# Relative plot
plotting.plot_DDGs(fe.graph,
                   target_name=f'{target}',
                   title=f'Relative binding energies - {target}',
                   figsize=5,
                   filename='./plot_relative.pdf'
                   )
# Absolute plot, with experimental data shifted to correct mean
experimental_mean_dg = np.asarray([node[1]["exp_DG"] for node in fe.graph.nodes(data=True)]).mean()
plotting.plot_DGs(fe.graph,
                  target_name=f'{target}',
                  title=f'Absolute binding energies - {target}',
                  figsize=5,
                  filename='./plot_absolute.pdf',
                  shift=experimental_mean_dg,
                  )


