""" Build simple graph from OpenEye or RDKit molecule object.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def fp_oe(atom):
    from openeye import oechem

    HYBRIDIZATION_OE = {
        oechem.OEHybridization_sp: torch.tensor(
            [1, 0, 0, 0, 0], dtype=torch.get_default_dtype()
        ),
        oechem.OEHybridization_sp2: torch.tensor(
            [0, 1, 0, 0, 0], dtype=torch.get_default_dtype()
        ),
        oechem.OEHybridization_sp3: torch.tensor(
            [0, 0, 1, 0, 0], dtype=torch.get_default_dtype()
        ),
        oechem.OEHybridization_sp3d: torch.tensor(
            [0, 0, 0, 1, 0], dtype=torch.get_default_dtype()
        ),
        oechem.OEHybridization_sp3d2: torch.tensor(
            [0, 0, 0, 0, 1], dtype=torch.get_default_dtype()
        ),
        oechem.OEHybridization_Unknown: torch.tensor(
            [0, 0, 0, 0, 0], dtype=torch.get_default_dtype()
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetDegree(),
                    atom.GetValence(),
                    atom.GetExplicitValence(),
                    atom.GetFormalCharge(),
                    atom.IsAromatic() * 1.0,
                    atom.GetIsotope(),  # TODO: is this a good idea?
                    oechem.OEAtomIsInRingSize(atom, 3) * 1.0,
                    oechem.OEAtomIsInRingSize(atom, 4) * 1.0,
                    oechem.OEAtomIsInRingSize(atom, 5) * 1.0,
                    oechem.OEAtomIsInRingSize(atom, 6) * 1.0,
                    oechem.OEAtomIsInRingSize(atom, 7) * 1.0,
                    oechem.OEAtomIsInRingSize(atom, 8) * 1.0,
                ],
                dtype=torch.float32,
            ),
            HYBRIDIZATION_OE[atom.GetHyb()],
        ],
        dim=0,
    )


def fp_rdkit(atom):
    from rdkit import Chem

    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: torch.tensor(
            [1, 0, 0, 0, 0], dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP2: torch.tensor(
            [0, 1, 0, 0, 0], dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3: torch.tensor(
            [0, 0, 1, 0, 0], dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D: torch.tensor(
            [0, 0, 0, 1, 0], dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
            [0, 0, 0, 0, 1], dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.S: torch.tensor(
            [0, 0, 0, 0, 0], dtype=torch.get_default_dtype(),
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetTotalDegree(),
                    atom.GetTotalValence(),
                    atom.GetExplicitValence(),
                    atom.GetFormalCharge(),
                    atom.GetIsAromatic() * 1.0,
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.get_default_dtype(),
            ),
            HYBRIDIZATION_RDKIT[atom.GetHybridization()],
        ],
        dim=0,
    )


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def from_openforcefield_mol(mol, use_fp=True):
    # initialize graph
    from rdkit import Chem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.n_atoms
    g.add_nodes(n_atoms)
    g.ndata["type"] = torch.Tensor(
        [[atom.atomic_number] for atom in mol.atoms]
    )

    h_v = torch.zeros(
        g.ndata["type"].shape[0], 100, dtype=torch.get_default_dtype()
    )

    h_v[
        torch.arange(g.ndata["type"].shape[0]),
        torch.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = torch.stack(
        [fp_rdkit(atom) for atom in mol.to_rdkit().GetAtoms()], axis=0
    )

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.bonds)
    bonds_begin_idxs = [bond.atom1_index for bond in bonds]
    bonds_end_idxs = [bond.atom2_index for bond in bonds]
    bonds_types = [bond.bond_order for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g


def from_oemol(mol, use_fp=True):
    from openeye import oechem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.NumAtoms()
    g.add_nodes(n_atoms)
    g.ndata["type"] = torch.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )

    h_v = torch.zeros(g.ndata["type"].shape[0], 100, dtype=torch.float32)

    h_v[
        torch.arange(g.ndata["type"].shape[0]),
        torch.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = torch.stack([fp_oe(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBgnIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndIdx() for bond in bonds]
    bonds_types = [bond.GetOrder() for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g


def from_rdkit_mol(mol, use_fp=True):
    from rdkit import Chem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.GetNumAtoms()
    g.add_nodes(n_atoms)
    g.ndata["type"] = torch.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )

    h_v = torch.zeros(g.ndata["type"].shape[0], 100, dtype=torch.float32)

    h_v[
        torch.arange(g.ndata["type"].shape[0]),
        torch.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = torch.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    bonds_types = [bond.GetBondType().real for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g
