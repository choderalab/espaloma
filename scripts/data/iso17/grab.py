import torch
import os
import numpy as np
import espaloma as esp
from openeye import oechem
from simtk import unit
from simtk.unit import Quantity

def run(idx, u_thres=0.1):
    # read xyz file
    energies = open("c7o2h10_md/%s.energy.dat" % idx, "r").readlines()[1:]
    energies = 0.037 * np.array([float(energy) for energy in energies])
    ifs = oechem.oemolistream()
    ifs.open("c7o2h10_md/%s.xyz" % idx)

    # get list of molecules
    mols = []
    for mol in ifs.GetOEGraphMols():
        mols.append(oechem.OEGraphMol(mol))
    assert len(mols) == len(energies)

    # read smiles
    smiles = [oechem.OEMolToSmiles(mol) for mol in mols]

    # find the reference smiles
    idx_ref = energies.argmin()
    smiles_ref = smiles[idx_ref]
    u_ref = energies.min()

    # pick the indices that are low in energy and have the same smiles string
    ok_idxs = [_idx for _idx in range(len(smiles)) if smiles[_idx] == smiles_ref]
    ok_idxs = [_idx for _idx in ok_idxs if energies[_idx] <= u_ref + u_thres]

    # filter based on indices
    ok_mols = [mols[_idx] for _idx in ok_idxs]
    ok_energies = [energies[_idx] for _idx in ok_idxs]

    ofs = oechem.oemolostream()
    ofs.open("mol.sdf")
    oechem.OEWriteMolecule(ofs, ok_mols[0])

    # put coordinates in the graph
    xs = [mol.GetCoords() for mol in ok_mols]
    xs = torch.stack(
            [
                torch.tensor(
                    [
                        Quantity(
                            x[_idx],
                            unit.angstrom,
                        ).value_in_unit(
                            unit.bohr,
                        )
                        for _idx in range(len(x))
                    ]
                ) for x in xs],
            dim=1,
    )

    us = torch.tensor(ok_energies)[None, :]

    from openforcefield.topology import Molecule
    g = esp.Graph(Molecule.from_openeye(mols[idx_ref], allow_undefined_stereo=True))
    g.nodes['n1'].data['xyz'] = xs
    g.nodes['g'].data['u_ref'] = us
    g = esp.data.md.subtract_nonbonded_force(g, "openff-1.2.0")
    print(g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].min())
    g.save("out/%s" % idx)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
