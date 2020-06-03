import espaloma as esp
import torch
import dgl

def run():
    ds = esp.data.ESOL()[:16].to_homogeneous_with_legacy_typing()
    len_tr = int(0.8 * len(ds))
    len_te = len(ds) - len_tr
    ds_tr, ds_te = torch.utils.data.random_split(
            ds,
            [len_tr, len_te])

    ds_tr = torch.utils.data.DataLoader(
            ds_tr,
            batch_size=4,
            shuffle=True,
            collate_fn=esp.data.utils.collate_fn)

    layer = esp.nn.dgl_legacy.gn()

    net = esp.nn.Sequential(
            layer,
            [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    opt = torch.optim.Adam(net.parameters(), 1e-3)

    for g in ds_tr:
        y_hat = net(g)
        y = g.legacy_type

        print(y_hat)
        print(y)


if __name__ == '__main__':
    run()
