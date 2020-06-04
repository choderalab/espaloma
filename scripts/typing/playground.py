import espaloma as esp
import torch

def run():
    # grab dataset
    esol = esp.data.esol(first=20)

    # do some typing
    typing = esp.graphs.legacy_force_field.LegacyForceField('gaff-1.81')
    esol.apply(typing, in_place=True) # this modify the original data

    # split
    # NOTE:
    # I don't like torch-generic splitting function as it requires
    # specifically the volume of each partition and it is inconsistent
    # with the specification of __getitem__ method
    ds_tr, ds_te = esol.split([4, 1]) 
    
    # get a loader object that views this dataset in some way
    # using this specific flag the dataset turns into an iterator
    # that outputs loss function, per John's suggestion
    loader = ds_tr.view('graph-typing-loss', batch_size=2)

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn('GraphConv')

    # define a representation
    representation = esp.nn.Sequential(
            layer,
            [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    # define a readout
    readout = esp.nn.readout.node_typing.NodeTyping(
            in_features=32,
            n_classes=100) # not too many elements here I think?

    # defien an optimizer
    opt = torch.optim.Adam(
            list(representation.parameters()) + list(readout.parameters()),
            1e-3)

    # train it !
    for _ in range(10):
        for g, loss_fn in loader:
            opt.zero_grad()
            g_hat = readout(representation(g))
            loss = loss_fn(g_hat)
            loss.backward()
            opt.step()
   

    # test it
    # NOTE:
    # one-hot for nn_typing, int for legacy_typing
    nn_typing = torch.cat(
            [readout(representation(g.homograph)).ndata['nn_typing'].argmax(dim=-1) for g in ds_te])

    legacy_typing = torch.cat(
            [g.ndata['legacy_typing'] for g in ds_te])


    print('Accuracy %s' % (torch.sum(torch.equal(nn_typing, legacy_typing) * 1.0) / len(ds_te)))

if __name__ == '__main__':
    run()
