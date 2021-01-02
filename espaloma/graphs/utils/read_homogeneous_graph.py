""" Build simple graph from OpenEye or RDKit molecule object.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import dgl.backend as F

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fp_oe(atom):
    from openeye import oechem

    HYBRIDIZATION_OE = {
        oechem.OEHybridization_sp: F.tensor(
            [1, 0, 0, 0, 0],
        ) * 1.0,
        oechem.OEHybridization_sp2: F.tensor(
            [0, 1, 0, 0, 0],
        ) * 1.0,
        oechem.OEHybridization_sp3: F.tensor(
            [0, 0, 1, 0, 0],
        ) * 1.0,
        oechem.OEHybridization_sp3d: F.tensor(
            [0, 0, 0, 1, 0],
        ) * 1.0,
        oechem.OEHybridization_sp3d2: F.tensor(
            [0, 0, 0, 0, 1],
        ) * 1.0,
        oechem.OEHybridization_Unknown: F.tensor(
            [0, 0, 0, 0, 0],
        ) * 1.0,
    }
    return F.cat(
        [
            F.tensor(
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
                dtype=F.float32,
            ),
            HYBRIDIZATION_OE[atom.GetHyb()],
        ],
        dim=0,
    )


def fp_rdkit(atom):
    from rdkit import Chem

    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: F.tensor(
            [1, 0, 0, 0, 0],
        ) * 1.0,
        Chem.rdchem.HybridizationType.SP2: F.tensor(
            [0, 1, 0, 0, 0],
        ) * 1.0,
        Chem.rdchem.HybridizationType.SP3: F.tensor(
            [0, 0, 1, 0, 0],
        ) * 1.0,
        Chem.rdchem.HybridizationType.SP3D: F.tensor(
            [0, 0, 0, 1, 0],
        ) * 1.0,
        Chem.rdchem.HybridizationType.SP3D2: F.tensor(
            [0, 0, 0, 0, 1],
        ) * 1.0,
        Chem.rdchem.HybridizationType.S: F.tensor(
            [0, 0, 0, 0, 0],
        ) * 1.0,
    }
    return F.cat(
        [
            F.tensor(
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
    g.ndata["type"] = F.tensor(
        [[atom.atomic_number] for atom in mol.atoms]
    )

    h_v = F.zeros(
        (g.ndata["type"].shape[0], 100,),
        F.float32,
        F.cpu(),
    )

    if F.backend_name == "pytorch":
        import torch
        h_v[
            torch.arange(g.ndata["type"].shape[0]),
            torch.squeeze(g.ndata["type"]).long(),
        ] = 1.0

    elif F.backend_name == "jax":
        import jax
        import jax.numpy as jnp
        h_v = jax.ops.index_update(
            h_v,
            jnp.concatenate(
                [
                    jnp.arange(g.ndata["type"].shape[0]),
                    jnp.squeeze(g.ndata["type"]),
                ]
            ),
            1.0,
        )

    h_v_fp = F.stack(
        [fp_rdkit(atom) for atom in mol.to_rdkit().GetAtoms()], dim=0
    )

    if use_fp == True:
        h_v = F.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.bonds)
    bonds_begin_idxs = [bond.atom1_index for bond in bonds]
    bonds_end_idxs = [bond.atom2_index for bond in bonds]
    bonds_types = [bond.bond_order for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = F.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g


def from_oemol(mol, use_fp=True):
    from openeye import oechem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.NumAtoms()
    g.add_nodes(n_atoms)
    g.ndata["type"] = F.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )

    h_v = F.zeros(g.ndata["type"].shape[0], 100, dtype=F.float32)

    h_v[
        F.arange(g.ndata["type"].shape[0]),
        F.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = F.stack([fp_oe(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = F.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBgnIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndIdx() for bond in bonds]
    bonds_types = [bond.GetOrder() for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = F.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g


def from_rdkit_mol(mol, use_fp=True):
    from rdkit import Chem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = mol.GetNumAtoms()
    g.add_nodes(n_atoms)
    g.ndata["type"] = F.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )

    h_v = F.zeros(g.ndata["type"].shape[0], 100, dtype=F.float32)

    h_v[
        F.arange(g.ndata["type"].shape[0]),
        F.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = F.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = F.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    bonds_types = [bond.GetBondType().real for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = F.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g
