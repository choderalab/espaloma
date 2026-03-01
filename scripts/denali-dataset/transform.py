import espaloma as esp

def run(in_path, out_path, u_threshold=0.1):
    g = esp.Graph.load(in_path)
    from espaloma.data.md import subtract_nonbonded_force
    g = subtract_nonbonded_force(g, subtract_charges=True)
    
    # get number of snapshots
    n_data = g.nodes['n1'].data['xyz'].shape[1]
    u_min = g.nodes['g'].data['u_ref'].min().item()

    print(n_data)

    # original indicies
    idxs = list(range(n_data))
    idxs = [idx for idx in idxs if g.nodes['g'].data['u_ref'][:, idx].item() < u_min + u_threshold]
    
    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, idxs, :]
    g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'][:, idxs]
    
    n_data = len(idxs)

    print(n_data)
    if n_data > 1:
        g.save(out_path)

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])



