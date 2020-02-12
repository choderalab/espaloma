""" Utilities for data pipeline.
"""
import torch
import dgl
from itertools import islice
import random

class BatchedDataset():
    def __init__(self, iterable, batch_size, n_batches_in_buffer=12, cache=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.n_batches_in_buffer= n_batches_in_buffer
        self.cache = cache

        if cache == True:
            self.finished = False
            self.cached_data = []

    def __iter__(self):
        if self.cache == True:
            if self.finished == True:
                return iter(self.cached_data)

        else:
            for x in self._iter():
                yield x

            self.finished = True


    def __iter__(self):
        iterable = self.iterable()

        while True:
            if self.n_batches_in_buffer != -1:
                try:
                    buffer = [ # grab some data from self.iterable
                        next(iterable) for _ in range(
                            self.n_batches_in_buffer * self.batch_size)]

                    random.shuffle(buffer)
                except:
                    break

                for idx in range(self.n_batches_in_buffer):
                    next_batch_list = buffer[
                        idx * self.batch_size : (idx+1) * self.batch_size]

                    # put graph and attributes in lists
                    gs = []
                    ys = []

                    for g, y in next_batch_list:
                        gs.append(g)
                        ys.append(y)

                    g_batched = dgl.batch(gs)
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

                g_batched = dgl.batch(gs)
                y_batched = torch.stack(ys, axis=0)

                if self.cache == True:
                    self.cached_data.append((g_batched, y_batched))

                yield g_batched, y_batched


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
