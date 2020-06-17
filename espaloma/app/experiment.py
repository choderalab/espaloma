# =============================================================================
# IMPORTS
# =============================================================================
import espaloma as esp
import abc
import torch
import copy

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Experiment(abc.ABC):
    """ Base class for espaloma experiment.

    """
    def __init__(self):
        super(Experiment, self).__init__()

class Train(Experiment):
    """ Train a model for a while.
    """
    def __init__(
            self,
            net,
            data,
            metrics=[esp.metrics.TypingCrossEntropy],
            optimizer=lambda net: torch.optim.Adam(net.parameters(), 1e-3),
            n_epochs=100,
            record_interval=1,
        ):
        super(Train, self).__init__()

        # bookkeeping
        self.net = net
        self.data = data
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.record_interval = record_interval
        self.states = {}

        # make optimizer
        if callable(optimizer):
            self.optimizer = optimizer(net)
        else:
            self.optimizer = optimizer

        # compose loss function
        def loss(g):
            _loss = 0.
            for metric in self.metrics:
                _loss += metric(g)
            return _loss

        self.loss = loss

    def train_once(self):
        """ Train the model for one batch.

        """
        for g in self.data: # TODO: does this have to be a single g?

            def closure():
                self.optimizer.zero_grad()
                loss = self.loss(g)
                loss.backward()
                return loss

            self.optimizer.step()

    def train(self):
        """ Train the model for multiple steps and
        record the weights once every `record_interval`

        """
        for epoch_idx in range(int(self.n_epochs)):
            self.train_once()

            # record when `record_interval` is hit
            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        # record final state
        self.states['final'] = copy.deepcopy(self.net.state_dict())

        return self.net


class Test(Experiment):
    """ Run sequences of tests on a trained model.

    """
    def __init__(
            self,
            net,
            data,
            states,
            metrics=[esp.metrics.TypingCrossEntropy],
            sampler=None):
        # bookkeeping
        self.net = net
        self.data = data
        self.states = states
        self.metrics = metrics
        self.sampler = sampler

    def test(self):
        """ Run test.

        """
        # loop through the metrics
        for metric in metrics:
            results[metric.__name__] = {}

        # make it just one giant graph
        g = data[0:-1]

        for state_name, state in self.states.items(): # loop through states
            # load the state dict
            self.net.load_state_dict(state)

            # loop through the metrics
            results[metric.__name__][state_name] = metric(
                self.net,
                g,
                sampler=self.sampler).detach().cpu().numpy()

        # point this to self
        self.results = results
        return dict(results)

class TrainAndTest(Experiment):
    """ Train a model and then test it.

    """
    def __init__(
        self,
        net,
        ds_tr,
        ds_te,
        metrics_tr=[esp.metrics.TypingCrossEntropy],
        metrics_te=[esp.metrics.TypingCrossEntropy],
        optimizer=lambda net: torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=100,
        record_interval=1
    ):

        # bookkeeping
        self.net = net
        self.ds_tr = ds_tr
        self.ds_te = ds_te
        self.optimizer = optimizer
        self.n_epochs = n_epochs

    def __str__(self):
        _str = ''
        _str += '# model'
        _str += '\n'
        _str += str(self.net)
        _str += '\n'
        if hasattr(self.net, 'noise_model'):
            _str += '# noise model'
            _str += '\n'
            _str += str(self.net.noise_model)
            _str += '\n'
        _str += '# optimizer'
        _str += '\n'
        _str += str(self.optimizer)
        _str += '\n'
        _str += '# n_epochs'
        _str += '\n'
        _str += str(self.n_epochs)
        _str += '\n'
        return _str

    def run(self):
        """ Run train and test.

        """
        train = Train(
            net=self.net,
            data=self.ds_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs
        )

        train.train()

        self.states = train.states

        test = Test(
            net=self.net,
            data=self.ds_te,
            metrics=self.metrics,
            states=self.states,
            sampler=self.sampler
        )

        test.test()

        self.results_te = test.results

        test = Test(
            net=self.net,
            data=self.ds_tr,
            metrics=self.metrics,
            states=self.states,
            sampler=self.sampler
        )

        test.test()

        self.results_tr = test.results

        return{'test': self.results_te, 'train': self.results_tr}
