import numpy as np

from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import TAGConv

from valence_targets import form_valence_targets

### models
def bent_identity(x):
    return (torch.sqrt(x ** 2 + 1) - 1) / 2 + x


class MLP(nn.Module):
    """fixed number of hidden units and hidden layers"""

    def __init__(self, in_features, out_features, activation=F.relu):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class TAG(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, k=2, activation=F.relu):
        """
        TODO: variable number of layers
        """
        super(TAG, self).__init__()
        self.layer1 = TAGConv(in_feats, h_feats, k, activation=activation)
        self.layer2 = TAGConv(h_feats, h_feats, k, activation=activation)
        self.layer3 = TAGConv(h_feats, num_classes, k, activation=activation)
        self.activation = activation
    
    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = self.activation(h)
        h = self.layer2(graph, h)
        h = self.activation(h)
        h = self.layer3(graph, h)
        return h


class ValenceModel(nn.Module):
    def __init__(self, node_representation, bond_readout, angle_readout, torsion_readout):
        super(ValenceModel, self).__init__()
        self.node_representation = node_representation
        self.bond_readout = bond_readout
        self.angle_readout = angle_readout
        self.torsion_readout = torsion_readout
    
    def forward(self, graph, inds):
        bond_inds, angle_inds, torsion_inds = inds['bonds'], inds['angles'], inds['torsions']
        node_reps = self.node_representation.forward(graph, graph.ndata['element'])
        
        # bonds
        a, b = node_reps[bond_inds[:,0]], node_reps[bond_inds[:,1]]
        bonds = self.bond_readout(torch.cat((a, b), dim=1)) + self.bond_readout(torch.cat((b, a), dim=1))
        
        # angles
        a, b, c = node_reps[angle_inds[:,0]], node_reps[angle_inds[:,1]], node_reps[angle_inds[:,2]]
        abc, cba = torch.cat((a, b, c), dim=1), torch.cat((c, b, a), dim=1)
        angles = self.angle_readout(abc) + self.angle_readout(cba)
        
        # torsions
        a, b, c, d = node_reps[torsion_inds[:,0]], node_reps[torsion_inds[:,1]], node_reps[torsion_inds[:,2]], node_reps[torsion_inds[:,3]]
        abcd, dcba = torch.cat((a, b, c, d), dim=1), torch.cat((d, c, b, a), dim=1)
        torsions = self.torsion_readout(abcd)
        
        return dict(bonds=bonds, angles=angles, torsions=torsions)


### define targets
from openforcefield.typing.engines.smirnoff import ForceField
forcefield = ForceField('openff_unconstrained-1.0.0.offxml')

def label_offmol(offmol):
    return forcefield.label_molecules(offmol.to_topology())[0]

elements = [1,3,6,7,8,9,15,16,17,19,35,53]
element_encoder = OneHotEncoder(sparse=False)
element_encoder.fit(np.array(elements).reshape(-1,1))

def offmol_to_dgl(offmol):
    graph = dgl.from_networkx(offmol.to_networkx())
    atomic_nums = [a.element.atomic_number for a in offmol.atoms]
    X = element_encoder.transform(np.array(atomic_nums).reshape(-1,1))
    graph.ndata['element'] = torch.Tensor(X)
    return graph


if __name__ == '__main__':
    # get input
    from openforcefield.topology import Molecule
    smi = 'COc1cc2nc(C)nc(N[C@H](C)c3ccc(cc3)c4ccccc4)c2cc1OC'
    offmol = Molecule.from_smiles(smi)
    graph = offmol_to_dgl(offmol)
    
    # define targets
    labeled_mol = label_offmol(offmol)
    inds, valence_targets = form_valence_targets(labeled_mol)

    # define model
    input_dim, hidden_dim, node_dim = graph.ndata['element'].shape[1], 128, 64
    activation = bent_identity

    node_rep = TAG(input_dim, hidden_dim, node_dim, k=2, activation=activation)
    bond_readout = MLP(node_dim * 2, valence_targets['bonds'].shape[1], activation=activation)
    angle_readout = MLP(node_dim * 3, valence_targets['angles'].shape[1], activation=activation)
    torsion_readout = MLP(node_dim * 4, valence_targets['torsions'].shape[1], activation=activation)
    valence_model = ValenceModel(node_rep, bond_readout, angle_readout, torsion_readout)

    # define loss, optimize
    optimizer = torch.optim.LBFGS(valence_model.parameters(), line_search_fn='strong_wolfe', history_size=10)

    def forward():
        return valence_model.forward(graph, inds)

    def compute_loss(preds, components=['bonds', 'angles', 'torsions']):
        return sum([F.mse_loss(preds[component], valence_targets[component]) for component in components])

    def closure():
        optimizer.zero_grad()
        preds = forward()
        loss = compute_loss(preds)
        loss.backward()
        return loss
    
    def detach_dict(d):
        d = dict()
        for key in d:
            d[key] = d[key].detach().numpy()
        return d
    
    optimizer = torch.optim.LBFGS(valence_model.parameters(), line_search_fn='strong_wolfe', history_size=10)
    
    preds = forward()
    predictions = [detach_dict(preds)]
    loss_traj = [compute_loss(preds).detach().numpy()]
    valence_model.train()
    from tqdm import tqdm
    trange = tqdm(range(200))
    for epoch in trange:
        loss = optimizer.step(closure)
        loss_traj.append(float(loss.detach().numpy()))
        predictions.append(detach_dict( forward()))
        trange.set_postfix(loss=loss.detach().numpy())
    
