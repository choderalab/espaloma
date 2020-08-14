#!/usr/bin/env python
# coding: utf-8

# In[128]:


import torch
import espaloma as esp

torch.autograd.set_detect_anomaly(True)
# In[145]:

def euler_method(qs, ps, q_grad, lr=1e-3):
    q = qs[-1]
    p = ps[-1]

    if q_grad is None:
        q_grad = torch.zeros_like(q)

    p = p.clone().add(lr * q_grad)
    q = q.clone().add(lr * p)
   
    ps.append(p)
    qs.append(q)
    
    return qs, ps


# In[146]:


class NodeRNN(torch.nn.Module):
    def __init__(self, input_size=32, units=128):
        super(NodeRNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=units,
            batch_first=True,
            bidirectional=True,
        )
        self.windows=48
        self.d = torch.nn.Linear(
            2 * units + self.windows,
            1
        )

    def apply_nodes_fn(self, x):
        h_rnn = self.rnn(x[:, None, :].repeat(1, self.windows, 1))[0]
       
        h_one_hot = torch.zeros(
            self.windows,
            self.windows
        ).scatter(
            1,
            torch.range(0, self.windows-1)[:, None].long(),
            1.0,
        )[None, :, :].repeat(x.shape[0], 1, 1)
        
        h = torch.cat(
            [
                h_rnn,
                h_one_hot,
            ],
            dim=-1
        )

        return self.d(h).squeeze(-1)

    def forward(self, g):
        g.apply_nodes(
            lambda node: {'lambs_': self.apply_nodes_fn(node.data['h'])},
            ntype='n2'
        )
        
        g.apply_nodes(
            lambda node: {'lambs_': self.apply_nodes_fn(node.data['h'])},
            ntype='n3'
        )
        
        return g
        


# In[167]:


class LambdaConstraint(torch.nn.Module):
    def __init__(self):
        super(LambdaConstraint, self).__init__()
        
    def forward(self, g):
        g.apply_nodes(
            lambda node: {'lambs': node.data['lambs_'].softmax(dim=-1).cumsum(dim=-1)},
            ntype='n2'
        )
        
        g.apply_nodes(
            lambda node: {'lambs': node.data['lambs_'].softmax(dim=-1).cumsum(dim=-1)},
            ntype='n3'
        )
        
        return g


# In[168]:


g = esp.Graph('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
esp.graphs.LegacyForceField('smirnoff99Frosst').parametrize(g)


# In[169]:


layer = esp.nn.dgl_legacy.gn()

representation = esp.nn.Sequential(
    layer,
    [32, 'tanh', 32, 'tanh', 32, 'tanh'],
)

readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=32,
    config=[32, 'tanh', 32],
    out_features={
        1: {'h': 32},
        2: {'h': 32},
        3: {'h': 32},
    }
)

node_rnn = NodeRNN()

lambda_constraint = LambdaConstraint()

net = torch.nn.Sequential(
    representation,
    readout,
    node_rnn,
    lambda_constraint,
)


# In[170]:


def f(x, idx, g):
    if idx == 0:
        return (x ** 2).sum(dim=(0, 2))
    
    if idx == 49:
        g.heterograph.nodes['n1'].data['xyz'] = x
        esp.mm.geometry.geometry_in_graph(g.heterograph)
        esp.mm.energy.energy_in_graph(g.heterograph, suffix='_ref')
        # print(g.nodes['n2'].data['u'].sum(dim=0) + g.nodes['n3'].data['u'].sum(dim=0))
        return 1e-10 * (g.nodes['n2'].data['u_ref'].sum(dim=0) + g.nodes['n3'].data['u_ref'].sum(dim=0))

    g.heterograph.nodes['n1'].data['xyz'] = x
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

    return 1e-10 * (g.nodes['n2'].data['u'].sum(dim=0) + g.nodes['n3'].data['u'].sum(dim=0))


# In[171]:


def loss(g):
    x = torch.randn(
            g.heterograph.number_of_nodes('n1'),
            128,
            3
    )

    x.requires_grad = True

    q = torch.zeros_like(x)

    xs = [x]
    qs = [q]
    
    works = 0.0
    
    for idx in range(1, 50):
        x = xs[-1]
        q = qs[-1]
    
        energy_old = f(x, idx-1, g)
        energy_new = f(x, idx, g)
        x_grad = torch.autograd.grad(
            energy_new.sum(),
            [x],
            create_graph=True
        )[0]

        xs, qs = euler_method(xs, qs, x_grad)
        
        works += energy_new - energy_old
        
    return works.sum()


# In[171]:


optimizer = torch.optim.SGD(net.parameters(), 1e-2, 1e-2)
for _ in range(1000):
    optimizer.zero_grad()
    net(g.heterograph)
    _loss = loss(g)
    _loss.backward(retain_graph=True)
    print(_loss)
    optimizer.step()


# In[140]:


from matplotlib import pyplot as plt
plt.plot(g.nodes['n2'].data['lambs_'][0].detach())


# In[ ]:




