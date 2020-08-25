#!/usr/bin/env python
# coding: utf-8

# # imports

# In[14]:


import torch
import espaloma as esp


# In[15]:


WINDOWS = 20


# # integrator
# We use a vanilla Euler method as our integrator.

# In[16]:


def velocity_verlot_integrator(xs, vs, closure, dt=1.0):
    x = xs[-1]
    v = vs[-1]

    energy_old = closure(x)

    a_old = -torch.autograd.grad(
        energy_old.sum(),
        [x],
        create_graph=True,
        retain_graph=True,
    )[0]

    x = x + v * dt + 0.5 * a_old * dt * dt

    energy_new = closure(x)

    a_new = -torch.autograd.grad(
        energy_new.sum(),
        [x],
        create_graph=True,
        retain_graph=True,
    )[0]

    v = v + 0.5 * a_old * dt + 0.5 * a_new * dt

    vs.append(v)
    xs.append(x)

    return xs, vs, energy_old, energy_new


# In[22]:


g = esp.Graph('C')
g = esp.graphs.LegacyForceField('smirnoff99Frosst').parametrize(g)


# In[23]:


layer = esp.nn.dgl_legacy.gn()

representation = esp.nn.Sequential(
    layer,
    [32, 'tanh', 32, 'tanh', 32, 'tanh'],
)

readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=32,
    config=[32, 'tanh', 32],
    out_features={
        1: {'epsilons': WINDOWS, 'sigmas': WINDOWS},
        2: {'ks': WINDOWS, 'eqs': WINDOWS},
        3: {'ks': WINDOWS, 'eqs': WINDOWS},
    }
)

net = torch.nn.Sequential(
    representation,
    readout,
    esp.mm.geometry.GeometryInGraph(),
    esp.mm.energy.EnergyInGraph(suffix='_ref'),
)


# In[24]:


particle_distribution = torch.distributions.normal.Normal(
    loc=torch.zeros(g.heterograph.number_of_nodes('n1'), 128, 3),
    scale=torch.ones(g.heterograph.number_of_nodes('n1'), 128, 3),
)


# In[25]:


def energy(g, idx):
    u2 = g.nodes['n2'].data['ks'][:, idx][:, None] * (
        g.nodes['n2'].data['x'] - g.nodes['n2'].data['eqs'][:, idx][:, None]
    ) ** 2
    
    u3 = g.nodes['n3'].data['ks'][:, idx][:, None] * (
        g.nodes['n3'].data['x'] - g.nodes['n3'].data['eqs'][:, idx][:, None]
    ) ** 2
    
    return u2.sum(dim=0) + u3.sum(dim=0)


# In[26]:


def simulation(net):
    x = torch.nn.Parameter(particle_distribution.sample())
    v = torch.zeros_like(x)
    
    xs = [x]
    vs = [v]
    
    for idx in range(1, WINDOWS):
        def closure(x):
            g.nodes['n1'].data['xyz'] = x
            net(g.heterograph)
            return energy(g, idx)
            
        xs, vs, _, __ = velocity_verlot_integrator(xs, vs, closure, 0.01)
        
    
    g.nodes['n1'].data['xyz'] = xs[-1]

    return g.nodes['g'].data['u_ref']


# In[ ]:


optimizer = torch.optim.Adam(net.parameters(), 1e-3)

for _ in range(100):
    optimizer.zero_grad()
    loss = simulation(net).sum()
    loss.backward(retain_graph=True)
    print(loss)
    optimizer.step()


# In[33]:


for _ in range(100):
    optimizer.zero_grad()
    loss = simulation(net).sum()
    loss.backward(retain_graph=True)
    print(loss)
    optimizer.step()
    
    


# In[32]:


g.nodes['n2'].data['x']


# In[28]:





# In[ ]:




