import hgfp
import argparse
import torch
import numpy as np
import itertools


def run(args):
    net = getattr(
        hgfp.models,
        args.model.lower()).Net(
            args.config)

    ds = getattr(
        hgfp.data,
        args.data.lower()).param.batched(
            num=args.size,
            batch_size=args.batch_size)


    ds_all = dgl.batch_hetero(list(itertools.chain.from_iterable(
        [dgl.unbatch_hetero(g) for g in ds()])))

    mean_and_std_dict = {}

    for term in ['atom', 'bond', 'angle']:
        for param in ['k', 'eq']:
            x = ds_all.nodes[term].data[param + '_ref']
            mean = np.mean(x.numpy())
            std = np.std(x.numpy())
            mean_and_std_dict[term][param]['mean'] = mean
            mean_and_std_dict[term][param]['std'] = std


    ds_tr, ds_te, ds_vl = hgfp.data.utils.split(
        ds,
        args.n_batches_te,
        args.n_batches_vl)

    optimizer = getattr(
        torch.optim,
        args.optimizer)(
            net.parameters(),
            lr=args.learning_rate)

    loss_fn = getattr(
        torch.nn.functional,
        args.loss_fn)

    if args.report == True:
        from matplotlib import pyplot as plt
        import time
        from time import localtime, strftime
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        import os

        time_str = strftime("%Y-%m-%d_%H_%M_%S", localtime())
        os.mkdir(time_str)

        time0 = time.time()

        f_handle = open(time_str + '/report.md', 'w')
        f_handle.write(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        f_handle.write('\n')
        f_handle.write('===========================')
        f_handle.write('\n')

    for epoch in range(args.n_epochs):
        for g, u in ds_tr:
            u_hat = net(g)
            u = norm(u)


            loss = loss_fn(u, u_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.report == True:
                losses = np.concatenate([losses, [loss.detach().numpy()]], axis=0)

    if args.report == True:

        time1 = time.time()

        f_handle.write('# Model Summary\n')

        for arg in vars(args):
            f_handle.write(arg+ '=' + str(getattr(args, arg)))
            f_handle.write('\n')

        f_handle.write('\n')
        f_handle.write(str(net))
        f_handle.write('\n')

        f_handle.write('# Time Used \n')
        f_handle.write('%.2f' % (time1 - time0))
        f_handle.write('\n')
        f_handle.write('\n')

        f_handle.write('# Dataset Size')
        f_handle.write('\n')
        f_handle.write('Training samples: ')
        f_handle.write('\n')
        f_handle.write('Training: %s, Validation: %s, Test: %s' % (
            u_tr.shape[0],
            u_vl.shape[0],
            u_te.shape[0]))
        f_handle.write('\n')

        rmse_tr = (
            np.sqrt(
                mean_squared_error(
                    u_tr,
                    u_hat_tr)))

        rmse_te = (
            np.sqrt(
                mean_squared_error(
                    u_te,
                    u_hat_te)))

        rmse_vl = (
            np.sqrt(
                mean_squared_error(
                    u_vl,
                    u_hat_vl)))

        r2_tr = (
            r2_score(
                u_tr,
                u_hat_tr))

        r2_te = (
            r2_score(
                u_te,
                u_hat_te))

        r2_vl = (
            r2_score(
                u_vl,
                u_hat_vl))

        f_handle.write('# Performance')

        f_handle.write('\n')

        f_handle.write('{:<15}'.format('|'))
        f_handle.write('{:<15}'.format('|R2'))
        f_handle.write('{:<15}'.format('|RMSE')+ '|' + '\n')

        f_handle.write('{:<15}'.format('|' + '-' * 13))
        f_handle.write('{:<15}'.format('|' + '-' * 13))
        f_handle.write('{:<15}'.format('|' + '-' * 13))
        f_handle.write('|' + '\n')

        f_handle.write('{:<15}'.format('|TRAIN'))
        f_handle.write('{:<15}'.format('|%.2f' % r2_tr))
        f_handle.write('{:<15}'.format('|%.2f' % rmse_tr) + '|' + '\n')

        f_handle.write('{:<15}'.format('|VALIDATION'))
        f_handle.write('{:<15}'.format('|%.2f' % r2_vl))
        f_handle.write('{:<15}'.format('|%.2f' % rmse_vl) + '|' + '\n')

        f_handle.write('{:<15}'.format('|TEST'))
        f_handle.write('{:<15}'.format('|%.2f' % r2_te))
        f_handle.write('{:<15}'.format('|%.2f' % rmse_te) + '|' + '\n')

        f_handle.write('\n')

        f_handle.write('<div align="center"><img src="loss.jpg" width="600"></div>')
        f_handle.write('\n')
        f_handle.write('<div align="center"><img src="RMSE.jpg" width="600"></div>')
        f_handle.write('\n')
        f_handle.write('<div align="center"><img src="R2.jpg" width="600"></div>')

        f_handle.close()

    torch.save(net.state_dict(), time_str + '/model')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='GCN')
    parser.add_argument('--config', required=True, nargs='*')
    parser.add_argument('--hetero', default=False, type=bool, nargs='?')
    parser.add_argument('--data', default='QM9')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_epochs', default='30', type=int)
    parser.add_argument('--size', default=-1, type=int)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--loss_fn', default='mse_loss')

    parser.add_argument('--n_batches_te', default=10, type=int)
    parser.add_argument('--n_batches_vl', default=10, type=int)
    parser.add_argument('--report', default=True)

    args = parser.parse_args()
    run(args)
