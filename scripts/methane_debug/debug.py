# In[0]
import dgl
import numpy as np
import torch
import espaloma as esp


# In[74]:


g = esp.Graph('C')


# In[75]:


forcefield = esp.graphs.legacy_force_field.LegacyForceField(
    "smirnoff99Frosst"
)

forcefield.parametrize(g)


# In[76]:


from espaloma.data.md import MoleculeVacuumSimulation
simulation = MoleculeVacuumSimulation(
    n_samples=100,
    n_steps_per_sample=10,
)
simulation.run(g)


# In[77]:


g.heterograph.nodes['n1'].data['xyz'].mean(dim=1)


# In[78]:


representation = esp.nn.baselines.FreeParameterBaseline(g_ref=g.heterograph)

net = torch.nn.Sequential(
        representation, 
        esp.mm.geometry.GeometryInGraph(),
        esp.mm.energy.EnergyInGraph(), # predicted energy -> u
        esp.mm.energy.EnergyInGraph(suffix='_ref') # reference energy -> u_ref,
)

optimizer = torch.optim.Adam(
    net.parameters(),
    0.1,
)

# optimizer = torch.optim.LBFGS(
#     net.parameters(),
#     0.1,
#     line_search_fn='strong_wolfe',
# )


# In[79]:


states = []
losses = []


# In[88]:


for name, param in net.named_parameters():
    print(name)
    print(param)


# In[80]:


for _ in range(1000):
    optimizer.zero_grad()
    
    def l():
        net(g.heterograph)
        
        
        loss = torch.nn.MSELoss()(
            g.nodes['n2'].data['u_ref'],
            g.nodes['n2'].data['u'],
        )

        loss = loss.sum()
    
        
        loss.backward()
        
        print(loss)
        return loss
    
    optimizer.step(l)


# In[83]:


eqs


# In[81]:


plt.plot(losses)


# In[69]:


ks = np.array([state['n2_k'].flatten() for state in states])
eqs = np.array([state['n2_eq'].flatten() for state in states])


# In[87]:


eqs.std(axis=0)


# In[73]:


for idx in range(8):
    plt.plot(np.diff(ks[:, idx]))
    


# In[55]:


plt.plot(ks, label='k')
plt.plot(eqs, label='eq')


# In[47]:


from matplotlib import pyplot as plt
plt.scatter(
    g.nodes['n2'].data['u_ref'].detach(),
    g.nodes['n2'].data['u'].detach()
)

plt.xlabel('ref')
plt.ylabel('pred')


# In[ ]:




