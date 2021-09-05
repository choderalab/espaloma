#!/usr/bin/env python
# coding: utf-8

# In[44]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import espaloma as esp
import torch
import numpy as np


# In[45]:


from simtk import unit
GAS_CONSTANT = 8.31446261815324 * unit.joule / (unit.kelvin * unit.mole)
GAS_CONSTANT = GAS_CONSTANT.value_in_unit(
    esp.units.ENERGY_UNIT / (unit.kelvin)
)
kT = GAS_CONSTANT * 300


# In[46]:


WINDOWS = 50


# In[47]:


def leapfrog(xs, vs, closure, dt=1.0):
    x = xs[-1]
    v = vs[-1]

    x = x + v * dt

    energy_old = closure(x)

    a = -torch.autograd.grad(
        energy_old.sum(),
        [x],
        create_graph=True,
        retain_graph=True,
    )[0]

    v = v + a * dt

    x = x + v * dt

    vs.append(v)
    xs.append(x)

    return xs, vs


# In[48]:


gs = esp.data.dataset.GraphDataset([esp.Graph('C' * idx) for idx in range(1, 3)])
gs.apply(
    esp.graphs.LegacyForceField('smirnoff99Frosst-1.1.0').parametrize,
    in_place=True,
)
ds = gs.view(batch_size=len(gs))


# In[49]:


layer = esp.nn.dgl_legacy.gn()

representation = esp.nn.Sequential(
    layer,
    [32, 'leaky_relu', 128, 'leaky_relu', 128, 'leaky_relu'],
)

readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=128,
    config=[128, 'leaky_relu', 128, 'leaky_relu'],
    out_features={
        1: {'epsilons': WINDOWS, 'sigma': 1},
        2: {'ks': WINDOWS, 'eqs': WINDOWS},
        3: {'ks': WINDOWS, 'eqs': WINDOWS},
    }
)


net = torch.nn.Sequential(
    representation,
    readout,
)

realize = torch.nn.Sequential(
    esp.mm.geometry.GeometryInGraph(),
    esp.mm.energy.EnergyInGraph(suffix='_ref', terms=['n2', 'n3']),
)


# In[50]:


def closure(x, idx, g):
    with g.local_scope():
        g.nodes['n1'].data['xyz'] = x

        if idx != -1:

            g.nodes['n2'].data['eq_ref'] = g.nodes['n2'].data['eqs'][:, idx][:, None]
            g.nodes['n2'].data['k_ref'] = g.nodes['n2'].data['ks'][:, idx][:, None]

            g.nodes['n3'].data['eq_ref'] = g.nodes['n3'].data['eqs'][:, idx][:, None]
            g.nodes['n3'].data['k_ref'] = g.nodes['n3'].data['ks'][:, idx][:, None]

        realize(g)
        return g.nodes['g'].data['u_ref']


# In[51]:


def simulation(net, g):
    with g.local_scope():
        net(g)

        particle_distribution = torch.distributions.normal.Normal(
            loc=torch.zeros(g.number_of_nodes('n1'), 128, 3),
            scale=g.nodes['n1'].data['sigma'][:, :, None].repeat(1, 128, 3).exp()
        )

        #normal_distribution = torch.distributions.normal.Normal(0, 1.0)

        x = torch.nn.Parameter(
            particle_distribution.rsample()
        )

        v = torch.zeros_like(x)

        xs = [x]
        vs = [v]


        for idx in range(1, WINDOWS):

            xs, vs = leapfrog(xs, vs, lambda x: closure(x, idx, g=g), 1e-2)

        return xs, vs, particle_distribution


# In[53]:


optimizer = torch.optim.Adam(net.parameters(), 1e-3)
normal_distribution = torch.distributions.normal.Normal(0, 1.0)

for _ in range(10000):
    for g in ds:
        optimizer.zero_grad()

        xs, vs, particle_distribution = simulation(net, g)

        energy = closure(xs[-1], idx=-1, g=g).sum()

        log_p = -energy/kT + normal_distribution.log_prob(vs[-1]).sum()

        log_q = normal_distribution.log_prob(vs[0]).sum() + particle_distribution.log_prob(xs[0]).sum()

        loss = -log_p + log_q

        loss.backward()

        print(loss, energy)

        optimizer.step()


# In[ ]:


torch.save(
    net.state_dict(),
    'net.th'
)


# In[41]:


g = esp.Graph('CC')


# In[42]:


xs, vs, particle_distribution = simulation(net, g=g.heterograph)


# In[43]:


import nglview as nv
from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import AllChem

conf_idx = 1

mol = g.mol.to_rdkit()
AllChem.EmbedMolecule(mol)
conf = mol.GetConformer()

xs, vs, particle_distribution = simulation(net, g=g.heterograph)
x = xs[-1]


for idx_atom in range(mol.GetNumAtoms()):
    conf.SetAtomPosition(
        idx_atom,
        Point3D(
            float(x[idx_atom, conf_idx, 0]),
            float(x[idx_atom, conf_idx, 1]),
            float(x[idx_atom, conf_idx, 2]),
        ))

nv.show_rdkit(mol)


# In[ ]:
