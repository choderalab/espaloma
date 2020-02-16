import hgfp
import argparse
import torch
import numpy as np


def run(args):
    net = getattr(
        hgfp.models,
        args.model.lower()).Net(
            args.config)

    ds = getattr(
        hgfp.data,
        args.data.lower()).df.batched(
            num=args.size,
            batch_size=args.batch_size,
            cache=args.cache,
            n_batches_in_buffer=args.n_batches_in_buffer,
            hetero=args.hetero)

    ds_mean, ds_std = getattr(
        hgfp.data,
        args.data.lower()).df.mean_and_std()

    def unnorm(x):
        return x * ds_std + ds_mean

    def norm(x):
        return (x - ds_mean) / ds_std

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

        losses = np.array([0.])
        rmse_vl = []
        r2_vl = []
        rmse_tr = []
        r2_tr = []
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
            net.eval()
            u_tr = np.array([0.])
            u_hat_tr = np.array([0.])

            u_vl = np.array([0.])
            u_hat_vl = np.array([0.])

            with torch.no_grad():
                for g, u in ds_tr:
                    u_hat = unnorm(net(g))
                    u_tr = np.concatenate([u_tr, u.detach().numpy()], axis=0)
                    u_hat_tr = np.concatenate([u_hat_tr, u_hat.detach().numpy()], axis=0)

                for g, u in ds_te:
                    u_hat = unnorm(net(g))
                    u_vl = np.concatenate([u_vl, u.detach().numpy()], axis=0)
                    u_hat_vl = np.concatenate([u_hat_vl, u_hat.detach().numpy()], axis=0)

            u_tr = u_tr[1:]
            u_vl = u_vl[1:]
            u_hat_tr = u_hat_tr[1:]
            u_hat_vl = u_hat_vl[1:]

            rmse_tr.append(
                np.sqrt(
                    mean_squared_error(
                        u_tr,
                        u_hat_tr)))

            rmse_vl.append(
                np.sqrt(
                    mean_squared_error(
                        u_vl,
                        u_hat_vl)))

            r2_tr.append(
                r2_score(
                    u_tr,
                    u_hat_tr))

            r2_vl.append(
                r2_score(
                    u_vl,
                    u_hat_vl))

            plt.style.use('fivethirtyeight')
            plt.figure()
            plt.plot(rmse_tr[1:], label=r'$RMSE_\mathtt{TRAIN}$')
            plt.plot(rmse_vl[1:], label=r'$RMSE_\mathtt{VALIDATION}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(time_str + '/RMSE.jpg')
            plt.close()
            plt.figure()
            plt.plot(r2_tr[1:], label=r'$R^2_\mathtt{TRAIN}$')
            plt.plot(r2_vl[1:], label=r'$R^2_\mathtt{VALIDATION}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(time_str + '/R2.jpg')
            plt.close()
            plt.figure()
            plt.plot(losses[10:])
            plt.tight_layout()
            plt.savefig(time_str + '/loss.jpg')
            plt.close()

    if args.report == True:

        time1 = time.time()

        f_handle.write('# Model Summary\n')

        for arg in vars(args):
            f_handle.write(arg+ '=' + str(getattr(args, arg)))
            f_handle.write('\n')

        f_handle.write('\n')
        f_handle.write(str(net))
        f_handle.write('\n')
        f_handle.write('\n')

        f_handle.write('# Time Used \n')
        f_handle.write('%.2f' % (time1 - time0))
        f_handle.write('\n')
        f_handle.write('\n')

        np.save(time_str + '/loss', losses)
        np.save(time_str + '/rmse_tr', rmse_tr)
        np.save(time_str + '/rmse_vl', rmse_vl)
        np.save(time_str + '/r2_tr', r2_tr)
        np.save(time_str + '/r2_vl', r2_vl)

        f_handle.write('# Performance')

        u_tr = np.array([0.])
        u_hat_tr = np.array([0.])

        u_te = np.array([0.])
        u_hat_te = np.array([0.])

        u_vl = np.array([0.])
        u_hat_vl = np.array([0.])

        for g, u in ds_tr:

            u_tr = np.concatenate([u_tr, u.detach().numpy()], axis=0)
            u_hat_tr = np.concatenate([u_hat_tr, unnorm(net(g)).detach().numpy()], axis=0)

        for g, u in ds_vl:
            u_vl = np.concatenate([u_te, u.detach().numpy()], axis=0)
            u_hat_vl = np.concatenate([u_hat_vl, unnorm(net(g)).detach().numpy()], axis=0)

        for g, u in ds_te:
            u_te = np.concatenate([u_te, u.detach().numpy()], axis=0)
            u_hat_te = np.concatenate([u_hat_te, unnorm(net(g)).detach().numpy()], axis=0)


        np.save(time_str + '/u_tr', u_tr)
        np.save(time_str + '/u_te', u_te)
        np.save(time_str + '/u_vl', u_vl)

        np.save(time_str + '/u_hat_tr', u_hat_tr)
        np.save(time_str + '/u_hat_vl', u_hat_vl)
        np.save(time_str + '/u_hat_te', u_hat_te)

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


        f_handle.write('\n')
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
    parser.add_argument('--n_batches_in_buffer', default=12, type=int)
    parser.add_argument('--cache', default=True)
    parser.add_argument('--n_batches_te', default=10, type=int)
    parser.add_argument('--n_batches_vl', default=10, type=int)
    parser.add_argument('--report', default=True)

    args = parser.parse_args()
    run(args)
