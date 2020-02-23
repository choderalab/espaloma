"""Fetch molecules and assigned types from http://www.ccl.net/cca/data/parm_at_Frosst/ ,
define a generator that yields (openff_molecule, one_hot_atom_types) pairs."""

import tarfile
from os.path import exists

import numpy as np
from openforcefield.topology import Molecule

fname = 'parm_at_Frosst.tgz'
url = 'http://www.ccl.net/cca/data/parm_at_Frosst/parm_at_Frosst.tgz'

# download if we haven't already
if not exists(fname):
    print('Downloading {} from {}...'.format(fname, url))
    import urllib.request

    urllib.request.urlretrieve(url, fname)

# extract zinc and parm@frosst atom types
archive = tarfile.open(fname)

zinc_file = archive.extractfile('parm_at_Frosst/zinc.sdf')
zinc_p_f_types_file = archive.extractfile('parm_at_Frosst/zinc_p_f_types.txt')

zinc_p_f_types = [l.strip() for l in zinc_p_f_types_file.readlines()]
zinc_mols = Molecule.from_file(
    zinc_file,
    file_format='sdf',
    allow_undefined_stereo=True)

archive.close()

# convert types from strings to ints, for one-hot encoding
unique_types = sorted(list(set(zinc_p_f_types)))
np.save('p_f_types.npy', unique_types)
n_types = len(unique_types)
type_to_int = dict(zip(unique_types, range(len(unique_types))))
type_ints = np.array([type_to_int[t] for t in zinc_p_f_types])


# define generators
def zinc_p_f_atom_types_generator():
    """generate (openforcefield.topology.Molecule, np.array) pairs"""
    current_index = 0
    for mol in zinc_mols:
        y = np.zeros((mol.n_atoms, n_types))
        for i in range(mol.n_atoms):
            y[i, type_ints[current_index]] = 1
            current_index += 1
        yield (mol, y)


# TODO: zinc_am1bcc_atom_types
# TODO: zinc_am1bcc_bond_types

# TODO: mmff94_p_f_atom_types
# TODO: mmff94_am1bcc_atom_types
# TODO: mmff94_am1bcc_bond_types


if __name__ == '__main__':
    from tqdm import tqdm

    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    radius = 4
    fpSize = 256
    morgan_generator = GetMorganGenerator(radius=radius, fpSize=fpSize)


    def compute_atom_centered_morgan_fingerprints(rdmol, morgan_generator, fpSize):
        n_atoms = rdmol.GetNumAtoms()
        fingerprints = np.zeros((n_atoms, fpSize), dtype=int)

        for i in range(rdmol.GetNumAtoms()):
            fingerprint = morgan_generator.GetCountFingerprint(rdmol, fromAtoms=[i])
            for (key, val) in fingerprint.GetNonzeroElements().items():
                fingerprints[i, key] = val
        return fingerprints


    all_fingerprints = []
    all_atom_labels = []
    import numpy as np
    from tqdm import tqdm

    problem_mols = []
    errors = []
    import rdkit

    print('converting to rdkit and generating atom-centered fingerprints')
    for (mol, y) in tqdm(zinc_p_f_atom_types_generator()):
        rdmol = None
        try:
            rdmol = mol.to_rdkit()
            assert (type(mol.to_rdkit()) == rdkit.Chem.rdchem.Mol)
            all_fingerprints.append(compute_atom_centered_morgan_fingerprints(rdmol, morgan_generator, fpSize))
            all_atom_labels.append(np.argmax(y, axis=1))
        except Exception as e:
            errors.append(e)
            print('problem encountered!')
            problem_mols.append(mol)

    # checkpointing
    # from pickle import dump
    # print('saving...')
    # with open('parm_at_frosst_fp_task.pkl', 'wb') as f:
    #    dump((all_fingerprints, all_atom_labels), f)
    # print('done!')

    # from pickle import load

    # with open('parm_at_frosst_fp_task.pkl', 'rb') as f:
    #    all_fingerprints, all_atom_labels = load(f)

    from sklearn.ensemble import RandomForestClassifier

    mol_inds = np.arange(len(all_fingerprints))
    train_fraction = 0.8
    split_ind = int(train_fraction * len(mol_inds))

    np.random.shuffle(mol_inds)

    train, validate = mol_inds[:split_ind], mol_inds[split_ind:]

    X = np.vstack([all_fingerprints[i] for i in train])
    y = np.hstack([all_atom_labels[i] for i in train])

    X_validate = np.vstack([all_fingerprints[i] for i in validate])
    y_validate = np.hstack([all_atom_labels[i] for i in validate])

    print('X.shape', X.shape)
    print('y.shape', y.shape)

    print('classifyin...')
    params = dict(n_estimators=50, verbose=1, max_samples=10000)  # , class_weight='balanced_subsample')
    hashed_params = hash(tuple(params.keys()) + tuple(params.values()))
    clf = RandomForestClassifier(**params)

    clf.fit(X, y)
    print('done!')

    print('train score', clf.score(X, y))
    print('validation score', clf.score(X_validate, y_validate))

    # if you want to save the model
    # from pickle import dump
    # clf_pickle_path = 'parm_at_frosst_clf_{}.pkl'.format(hashed_params)

    # with open(clf_pickle_path, 'wb') as f:
    #    dump((clf, params), f)

    # unique_types = list(map(str, np.load('p_f_types.npy')))

    y_pred = clf.predict(X)
    from sklearn.metrics import confusion_matrix

    C_train = confusion_matrix(y, y_pred)  # , labels=unique_atoms)
    np.save('p_f_confusion_matrix_train.npy', C_train)

    # C_train = confusion_matrix(y, y_pred)#, labels=unique_atoms)
    # np.save('p_f_confusion_matrix_train.npy', C_train)

    # sort and label the bits...

    import matplotlib.pyplot as plt

    plt.imshow(C_train, cmap='Blues')
    plt.colorbar()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # subtract off the diagonal (leaving only the incorrect predictions)
    C_train_off_diagonal = C_train - np.diag(np.diag(C_train))
    plt.imshow(C_train_off_diagonal, cmap='Blues')
    plt.colorbar()
    plt.title('confusion matrix minus diagonal')
    plt.savefig('confusion_matrix_minus_diagonal.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('total number of mistakes (train set): {}'.format(C_train_off_diagonal.sum()))
    print('total number of predictions (train set): {}'.format(C_train.sum()))

    # atoms for which we make the most mistakes
    num_atoms = C_train.sum(1)
    num_mistakes = C_train_off_diagonal.sum(1)
    inds = np.argsort(-num_mistakes)
    print('training set mistakes:')
    print('atom_type\t#mistakes\t#atoms')
    for ind in inds[:10]:
        print('{}\t{}\t{}'.format(
            unique_types[ind], num_mistakes[ind], num_atoms[ind]))

    # pairs we most often confuse for each other
    flat_inds = np.argsort(-C_train_off_diagonal.flatten())
    print('training set mistakes:')
    print('true\tpred\t#occurrences')
    for ind in flat_inds[:10]:
        true, pred = np.unravel_index(ind, C_train_off_diagonal.shape)
        print('{}\t{}\t{}'.format(
            unique_types[true],
            unique_types[pred],
            C_train_off_diagonal[true, pred])
        )
