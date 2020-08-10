#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
import espaloma as esp


# In[116]:


class EulerIntegrator(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, m=0.1):
        defaults = dict(
            lr=lr,
            m=m,
        )
        super(EulerIntegrator, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for q in group['params']:
                if q.grad is None:
                    continue

                state = self.state[q]
                if len(state) == 0:
                    state['p'] = torch.zeros_like(q)

                state['p'].add_(q.grad, alpha=-group['lr']*group['m'])
                q.add_(state['p'], alpha=group['lr'])

        return loss


# In[117]:


g = esp.Graph('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
esp.graphs.LegacyForceField('smirnoff99Frosst').parametrize(g)


# In[118]:


layer = esp.nn.dgl_legacy.gn()

representation = esp.nn.Sequential(
    layer,
    [32, 'tanh', 32, 'tanh', 32, 'tanh'],
)

readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=32,
    config=[32, 'tanh', 32],
    out_features={
        1: {'lambs': 98},
        2: {'lambs': 98},
        3: {'lambs': 98},
    }
)

net = torch.nn.Sequential(
    representation,
    readout,
)


# In[139]:


def f(x, idx):
    print(idx)
    if idx == 0:
        return (x ** 2).sum(dim=(0, 2))
    
    if idx == 99:
        g.nodes['n1'].data['xyz'] = x
        esp.mm.geometry.geometry_in_graph(g.heterograph)
        esp.mm.energy.energy_in_graph(g.heterograph, suffix='_ref')
        # print(g.nodes['n2'].data['u'].sum(dim=0) + g.nodes['n3'].data['u'].sum(dim=0))
        return g.nodes['n2'].data['u_ref'].sum(dim=0) + g.nodes['n3'].data['u_ref'].sum(dim=0)

    g.nodes['n1'].data['xyz'] = x
    esp.mm.geometry.geometry_in_graph(g.heterograph)
    esp.mm.energy.energy_in_graph(g.heterograph, suffix='_ref')

    g.heterograph.apply_nodes(
        lambda node: {'u': node.data['u_ref'] * node.data['lambs'][:, idx-1][:, None]},
        ntype='n2'
    )

    g.heterograph.apply_nodes(
        lambda node: {'u': node.data['u_ref'] * node.data['lambs'][:, idx-1][:, None]},
        ntype='n3'
    )

    return g.nodes['n2'].data['u'].sum(dim=0) + g.nodes['n3'].data['u'].sum(dim=0)


# In[140]:


def loss():
    x = torch.autograd.Variable(
        torch.randn(
            g.heterograph.number_of_nodes('n1'),
            128,
            3
        )
    )
    
    sampler = EulerIntegrator([x], 1e-1)
    
    works = 0.0
    
    net(g.heterograph)
    
    for idx in range(1, 100):
        sampler.zero_grad()
        energy_old = f(x, idx-1)
        energy_new = f(x, idx)
        energy_new.sum().backward(create_graph=True)
        sampler.step()
        works += energy_new - energy_old
        
    return works.sum()


# In[141]:


optimizer = torch.optim.Adam(net.parameters(), 1e-5)
for _ in range(1000):
    optimizer.zero_grad()
    _loss = loss()
    _loss.backward()
    print(_loss)
    optimizer.step()


# In[ ]:




