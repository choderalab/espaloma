import torch
from openff.toolkit.topology import Molecule

import espaloma as esp
from .symmetry import ParameterizedSystem, Indices


def compute_bonds(
    xyz: torch.Tensor, params: ParameterizedSystem, indices: Indices
) -> torch.Tensor:
    a, b = xyz[:, indices.bonds[:, 0]], xyz[:, indices.bonds[:, 1]]
    distance = esp.mm.geometry.distance(a, b)
    k, eq = params.bonds[:, 0], params.bonds[:, 1]
    return esp.mm.bond.harmonic_bond(distance, k, eq)


def compute_angles(
    xyz: torch.Tensor, params: ParameterizedSystem, indices: Indices
) -> torch.Tensor:
    a, b, c = (
        xyz[:, indices.angles[:, 0]],
        xyz[:, indices.angles[:, 1]],
        xyz[:, indices.angles[:, 2]],
    )
    angles = esp.mm.geometry.angle(a, b, c)
    k, eq = params.angles[:, 0], params.angles[:, 1]
    return esp.mm.angle.harmonic_angle(angles, k, eq)


def compute_propers(
    xyz: torch.Tensor, params: ParameterizedSystem, indices: Indices
) -> torch.Tensor:
    # TODO: reduce code duplication with compute_impropers
    a, b = xyz[:, indices.propers[:, 0]], xyz[:, indices.propers[:, 1]]
    c, d = xyz[:, indices.propers[:, 2]], xyz[:, indices.propers[:, 3]]
    dihedrals = esp.mm.geometry.dihedral(a, b, c, d)
    ks = params.propers
    return esp.mm.functional.periodic_fixed_phases(dihedrals, ks)


def compute_impropers(
    xyz: torch.Tensor, params: ParameterizedSystem, indices: Indices
) -> torch.Tensor:
    # TODO: reduce code duplication with compute_propers
    a, b = xyz[:, indices.impropers[:, 0]], xyz[:, indices.impropers[:, 1]]
    c, d = xyz[:, indices.impropers[:, 2]], xyz[:, indices.impropers[:, 3]]
    dihedrals = esp.mm.geometry.dihedral(a, b, c, d)
    ks = params.impropers
    return esp.mm.functional.periodic_fixed_phases(dihedrals, ks)


def compute_valence_energy(
    offmol: Molecule, xyz: torch.Tensor, params: ParameterizedSystem
) -> torch.Tensor:
    indices = Indices(offmol)
    harmonic_terms = compute_bonds(xyz, params, indices).sum(
        1
    ) + compute_angles(xyz, params, indices).sum(1)
    torsion_terms = compute_propers(xyz, params, indices).sum(
        1
    ) + compute_impropers(xyz, params, indices).sum(1)
    return harmonic_terms + torsion_terms
