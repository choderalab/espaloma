""" Utilities for data pipeline.
"""
import torch
import dgl
from itertools import islice
import random
import numpy as np

class BatchedDataset():
    def __init__(self, iterable, batch_size, n_batches_in_buffer=12, cache=False, hetero=False,
            cat_not_stack=False, debug=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.n_batches_in_buffer= n_batches_in_buffer
        self.cache = cache
        self.hetero=hetero
        self.cat_not_stack = cat_not_stack
        self.finished = False
        self.cached_data = []
        self.debug = debug

    def __iter__(self):

        if self.cache == True and self.finished == True:

            random.shuffle(self.cached_data)

            for x in self.cached_data:
                yield x


        else:

            for x in self._iter():
                yield x

            self.finished = True


    def _iter(self):
        iterable = self.iterable()

        while True:
            if self.n_batches_in_buffer != -1:

                if self.debug==True:
                    buffer = [ # grab some data from self.iterable
                        next(iterable) for _ in range(
                            self.n_batches_in_buffer * self.batch_size)]
                    random.shuffle(buffer)
 
                else: 

                    try:
                        buffer = [ # grab some data from self.iterable
                            next(iterable) for _ in range(
                                self.n_batches_in_buffer * self.batch_size)]
                        random.shuffle(buffer)
                    except:
                        break
                random.shuffle(buffer)


                for idx in range(self.n_batches_in_buffer):
                    next_batch_list = buffer[
                        idx * self.batch_size : (idx+1) * self.batch_size]

                    # put graph and attributes in lists
                    gs = []
                    ys = []

                    for g, y in next_batch_list:
                        gs.append(g)
                        ys.append(y)

                    if self.hetero is True:
                        g_batched = dgl.batch_hetero(gs)
                    else:
                        g_batched = dgl.batch(gs)

                    if self.cat_not_stack == True:
                        y_batched = torch.cat(ys, axis=0)
                    else:
                        y_batched = torch.stack(ys, axis=0)

                    if self.cache == True:
                        self.cached_data.append((g_batched, y_batched))

                    yield g_batched, y_batched

            else:
                try:
                    next_batch_list = [
                        next(iterable) for _ in range(self.batch_size)]
                except:
                    break

                # put graph and attributes in lists
                gs = []
                ys = []

                for g, y in next_batch_list:
                    gs.append(g)
                    ys.append(y)

                if self.hetero is True:
                    g_batched = dgl.batch_hetero(gs)
                else:
                    g_batched = dgl.batch(gs)

                y_batched = torch.stack(ys, axis=0)

                if self.cache == True:
                    self.cached_data.append((g_batched, y_batched))

                yield g_batched, y_batched


class BatchedParamGraph():
    def __init__(self, iterable, batch_size=32):
        self.iterable = iterable
        self.batch_size = batch_size

    def __iter__(self):
        # _iter = iter(self.iterable())
        _iter = self.iterable()

        while True:
            try:
                yield dgl.batch_hetero([next(_iter) for _ in range(self.batch_size)])
            except:
                break

def split(ds, n_batches_te=10, n_batches_vl=10):
    
    ds_iter = iter(ds)

    ds_te = [next(ds_iter) for _ in range(n_batches_te)]
    ds_vl = [next(ds_iter) for _ in range(n_batches_vl)]

    class ds_tr():
        def __iter__(self):
            ds_iter = iter(ds)
            [next(ds_iter) for _ in range(n_batches_te + n_batches_vl)]
            for x in ds_iter:
                yield x

    return ds_tr(), ds_te, ds_vl



def get_norm_dict(ds):

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
   
    return mean_and_std_dict

def get_norm_fn(mean_and_std_dict):
    def norm(g, mean_and_std_dict=mean_and_std_dict):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                g.apply_nodes(
                    lambda node: {param:
                        (node.data[param] - mean_and_std_dict[term][param]['mean'])/\
                        mean_and_std_dict[term][param]['std']},
                    ntype=term)
        return g

    def unnorm(g, mean_and_std_dict=mean_and_std_dict):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                g.apply_nodes(
                    lambda node: {param:
                        (node.data[param] * \
                        mean_and_std_dict[term][param]['std'] + mean_and_std_dict[term][param]['mean'])},
                    ntype=term)
        return g

    return norm, unnorm

def get_norm_fn_log_normal(mean_and_std_dict):
    import math

    for term in ['atom', 'bond', 'angle']:
        for param in ['k', 'eq']:

            exp_mu_plus_half_sigma_2 = mean_and_std_dict[term][param]['mean']
            exp_sigma_2 = mean_and_std_dict[term][param]['std'] ** 2 / (exp_mu_plus_half_sigma_2 ** 2) + 1
            sigma_2 = math.log(exp_sigma_2)
            sigma = sigma_2 ** 0.5
            mu = math.log(exp_mu_plus_half_sigma_2) - 0.5 * sigma_2

            mean_and_std_dict[term][param]['mu'] = mu
            mean_and_std_dict[term][param]['sigma'] = sigma
            
    def norm(g, mean_and_std_dict=mean_and_std_dict):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                mu = mean_and_std_dict[term][param]['mu']
                sigma = mean_and_std_dict[term][param]['sigma']

                g.apply_nodes(
                    lambda node: {param:
                        (torch.log(node.data[param]) - mu)/ sigma},
                    ntype=term)
        return g

    def unnorm(g, mean_and_std_dict=mean_and_std_dict):
        for term in ['atom', 'bond', 'angle']:
            for param in ['k', 'eq']:
                mu = mean_and_std_dict[term][param]['mu']
                sigma = mean_and_std_dict[term][param]['sigma']
                
                g.apply_nodes(
                    lambda node: {param:
                        torch.exp(node.data[param] * sigma + mu)},
                    ntype=term)
        return g

    return norm, unnorm
 
