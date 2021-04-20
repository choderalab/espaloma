import h5py
import espaloma as esp

def run(path, name):
    ds = h5py.File(path)[name]
    for key in ds:
        mol = ds[key]

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
