import espaloma as esp

def run(in_path, out_path, batch_size=32, u_threshold=0.05):
    g = esp.Graph.load(in_path)
    from espaloma.data.md import subtract_nonbonded_force
    g = subtract_nonbonded_force(g)
    
    # get number of snapshots
    n_data = g.nodes['n1'].data['xyz'].shape[1]
    print("n_data: ", n_data)
    u_min = g.nodes['g'].data['u_ref'].min().item()

    # original indicies
    idxs = list(range(n_data))
    idxs = [idx for idx in idxs if g.nodes['g'].data['u_ref'][:, idx].item() < u_min + u_threshold]
    n_data = len(idxs)

    import random
    # additional indicies
    additional_n_data = batch_size - (n_data % batch_size)
    additional_idxs = random.choices(idxs, k=additional_n_data)
    idxs += additional_idxs
    random.shuffle(idxs)
    assert len(idxs) % batch_size == 0


    # put data into chunks
    import copy
    n_chunks = len(idxs) // batch_size
    for idx_chunk in range(n_chunks):
        print("idx: ", idx_chunk)
        chunk_idxs = idxs[idx_chunk*batch_size : (idx_chunk+1)*batch_size]
        _g = copy.deepcopy(g)
        _g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, chunk_idxs, :]
        _g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, chunk_idxs, :]
        _g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'][:, chunk_idxs]
        _g.save(out_path+str(idx_chunk))

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])



