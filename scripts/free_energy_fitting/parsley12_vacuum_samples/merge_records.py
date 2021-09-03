from glob import glob

import numpy as np

fnames = glob('*.npy')
xyz = dict()

for fname in fnames:
    key = fname.split('.')[-2]
    xyz[key] = np.load(fname)

np.savez('freesolv_vacuum_samples.npz', **xyz)
