import hgfp
import argparse
import torch
import numpy as np
import itertools
import dgl


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


    ds_all = []
    for g in ds:
        ds_all += dgl.unbatch_hetero(g)

    ds_all = dgl.batch_hetero(ds_all)

    mean_and_std_dict = {}

    for term in ['atom', 'bond', 'angle']:
        mean_and_std_dict[term] = {}
        for param in ['k', 'eq']:
            mean_and_std_dict[term][param] = {}
            x = ds_all.nodes[term].data[param + '_ref']
            mean = np.mean(x.numpy())
            std = np.std(x.numpy())
            mean_and_std_dict[term][param]['mean'] = mean
            mean_and_std_dict[term][param]['std'] = std

    def norm(g):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                g.apply_nodes(
                    lambda node: {param + '_ref':
                        (node.data[param + '_ref'] - mean_and_std_dict[term][param]['mean'])/\
                        mean_and_std_dict[term][param]['std']},
                    ntype=term)
        return g

    def unnorm(g):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                g.apply_nodes(
                    lambda node: {param:
                        (node.data[param] * \
                        mean_and_std_dict[term][param]['std'] + mean_and_std_dict[term][param]['mean'])},
                    ntype=term)
        return g

    list(ds)

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

    def graph_loss(g, loss_fn=loss_fn):
        g = norm(g)
        return torch.sum(torch.stack([loss_fn(
            g.nodes[term].data[param + '_ref'],
            g.nodes[term].data[param]) for term in ['atom', 'bond', 'angle']\
                    for param in ['k', 'eq']]))

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
        f_handle.write(strftime(time_str))
        f_handle.write('\n')
        f_handle.write('===========================')
        f_handle.write('\n')

    for epoch in range(args.n_epochs):
        for g in ds_tr:
            g_ = net(g, return_graph=True)
            loss = graph_loss(g_)
            print(loss, flush=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    net.eval()
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
            len(list(ds_tr)) * args.batch_size,
            len(list(ds_vl)) * args.batch_size,
            len(list(ds_te)) * args.batch_size))

        f_handle.write('\n')


        res = {}

        for term in ['atom', 'bond', 'angle']:
            res[term] = {}
            for param in ['k', 'eq']:
                res[term][param] = {}
                for part in ['tr', 'vl', 'te']:
                    res[term][param][part] = {}
                    for label in ['true', 'pred']:
                        res[term][param][part][label] = np.array([0.])
                        res[term][param][part][label] = np.array([0.])

        for part, ds_ in {'tr':ds_tr, 'vl':ds_vl, 'te':ds_te}.items():
            for g in ds_:
                g_ = net(g, return_graph = True)
                g_ = unnorm(g_)
                for term in ['atom', 'bond', 'angle']:
                    for param in ['k', 'eq']:
                        res[term][param][part]['true'] = np.concatenate(
                        [
                            res[term][param][part]['true'],
                            g_.nodes[term].data[param + '_ref'].detach().numpy()
                        ])

                        res[term][param][part]['pred'] = np.concatenate(
                        [
                            res[term][param][part]['pred'],
                            g_.nodes[term].data[param].detach().numpy()
                        ])


        f_handle.write('# Performance')

        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                f_handle.write('\n')
                f_handle.write(term + '_' + param)
                f_handle.write('\n')
                y_tr = res[term][param]['tr']['true'][1:]
                y_hat_tr = res[term][param]['tr']['pred'][1:]

                y_vl = res[term][param]['vl']['true'][1:]
                y_hat_vl = res[term][param]['vl']['pred'][1:]

                y_te = res[term][param]['te']['true'][1:]
                y_hat_te = res[term][param]['te']['pred'][1:]


                np.save(time_str + '/' + term + '_' +param + '_y_tr.npy', y_tr)
                np.save(time_str + '/' + term + '_' + param + '_y_hat_tr.npy', y_hat_tr)

                np.save(time_str + '/' + term + '_' + param + '_y_vl.npy', y_vl)
                np.save(time_str + '/' + term + '_' + param + '_y_hat_vl.npy', y_hat_vl)

                np.save(time_str + '/' + term + '_' + param + '_y_te.npy', y_te)
                np.save(time_str + '/' + term + '_' + param + '_y_hat_te.npy', y_hat_te)

                rmse_tr = (
                    np.sqrt(
                        mean_squared_error(
                            y_tr,
                            y_hat_tr)))

                rmse_te = (
                    np.sqrt(
                        mean_squared_error(
                            y_te,
                            y_hat_te)))

                rmse_vl = (
                    np.sqrt(
                        mean_squared_error(
                            y_vl,
                            y_hat_vl)))

                r2_tr = (
                    r2_score(
                        y_tr,
                        y_hat_tr))

                r2_te = (
                    r2_score(
                        y_te,
                        y_hat_te))

                r2_vl = (
                    r2_score(
                        y_vl,
                        y_hat_vl))

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
