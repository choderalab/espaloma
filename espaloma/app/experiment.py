# =============================================================================
# IMPORTS
# =============================================================================
import abc
import copy
import torch

import espaloma as esp


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Experiment(abc.ABC):
    """Base class for espaloma experiment."""

    def __init__(self):
        super(Experiment, self).__init__()


class Train(Experiment):
    """Training experiment.

    Parameters
    ----------
    net : `torch.nn.Module`
        Neural networks that inputs graph representation and outputs
        parameterized or typed graph for molecular mechanics.

    data : `espaloma.data.dataset.Dataset`
        or `torch.utils.data.DataLoader`
        Dataset.

    metrics : `List` of `callable`
        List of loss functions to be used (summed) in training.

    optimizer : `torch.optim.Optimizer`
        Optimizer used for training.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        Interval at which states are recorded.

    Methods
    -------
    train_once : Train the network for exactly once.

    train : Execute `train_once` for `n_epochs` times and record states
        every `record_interval`.

    """

    def __init__(
        self,
        net,
        data,
        metrics=[esp.metrics.TypingCrossEntropy()],
        optimizer=lambda net: torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=100,
        record_interval=1,
        normalize=esp.data.normalize.ESOL100LogNormalNormalize,
        scheduler=None,
        device=torch.device("cpu"),
    ):
        super(Train, self).__init__()

        # bookkeeping
        self.device = device
        if isinstance(net, torch.nn.DataParallel):
            self.net = net
        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
            self.net = net
        else:
            self.net = net.to(self.device)
        self.data = data
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.record_interval = record_interval
        self.normalize = normalize()
        self.states = {}
        self.scheduler = scheduler

        # make optimizer
        if callable(optimizer):
            self.optimizer = optimizer(net)
        else:
            self.optimizer = optimizer

        # compose loss function
        def loss(g):
            _loss = 0.0
            for metric in self.metrics:
                _loss += metric(g)

            return _loss

        self.loss = loss

    def train_once(self):
        """Train the model for one batch."""
        for idx, g in enumerate(
            self.data
        ):  # TODO: does this have to be a single g?

            if isinstance(self.optimizer, torch.optim.LBFGS):
                retain_graph = True
            else:
                retain_graph = False

            g = g.to(self.device)
            self.net.train()

            def closure(g=g):
                self.optimizer.zero_grad()
                g = self.net(g)
                g = self.normalize.unnorm(g)

                loss = self.loss(g)
                loss.backward(retain_graph=retain_graph)
                if idx == 0:
                    if torch.isnan(loss).cpu().numpy().item() is True:
                        raise RuntimeError("Loss is Nan.")
                return loss

            loss = closure()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step(loss)

    def train(self):
        """Train the model for multiple steps and
        record the weights once every `record_interval`

        """

        for epoch_idx in range(int(self.n_epochs)):

            self.train_once()

            # record when `record_interval` is hit
            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        # record final state
        self.states["final"] = copy.deepcopy(self.net.state_dict())

        return self.net


class Test(Experiment):
    """Test experiment.

    Parameters
    ----------
    net : `torch.nn.Module`
        Neural networks that inputs graph representation and outputs
        parameterized or typed graph for molecular mechanics.

    data : `espaloma.data.dataset.Dataset`
        or `torch.utils.data.DataLoader`
        Dataset.

    metrics : `List` of `callable`
        List of loss functions to be used (summed) in training.


    """

    def __init__(
        self,
        net,
        data,
        states,
        metrics=[esp.metrics.TypingCrossEntropy()],
        normalize=esp.data.normalize.NotNormalize,
        sampler=None,
        device=torch.device("cpu"),  # it should cpu
    ):
        # bookkeeping
        self.device = device
        self.net = net.to(self.device)
        self.data = data
        self.states = states
        self.metrics = metrics
        self.sampler = sampler
        self.normalize = normalize()

    def test(self):
        """Run tests."""

        results = {}

        # loop through the metrics
        for metric in self.metrics:
            results[metric.__name__] = {}

        # NOTE: we are not doing this here since this will lead to OOM
        # from time to time
        # make it just one giant graph
        # g = list(self.data)
        # g = dgl.batch(g)
        # g = g.to(self.device)

        if self.states is None:
            self.states = {"final": None}

        for state_name, state in self.states.items():  # loop through states
            if state is not None:
                # load the state dict
                self.net.load_state_dict(state)

            self.net.eval()

            for metric in self.metrics:
                assert isinstance(metric, esp.metrics.Metric)
                input_fn, target_fn = metric.between

                inputs = []
                targets = []

                for g in self.data:
                    with g.local_scope():
                        g = g.to(self.device)
                        g_input = self.normalize.unnorm(self.net(g))
                        inputs.append(input_fn(g_input).detach())
                        targets.append(target_fn(g_input).detach())

                inputs = torch.cat(inputs, dim=0)
                targets = torch.cat(targets, dim=0)

                # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric.base_metric(inputs, targets).detach().cpu().numpy()
                )

        self.ref_g = self.normalize.unnorm(self.net(g)).to(
            torch.device("cpu")
        )

        for term in self.ref_g.ntypes:
            for param in self.ref_g.nodes[term].data.keys():
                g.nodes[term].data[param] = g.nodes[term].data[param].detach()

        # point this to self
        self.results = results


class TrainAndTest(Experiment):
    """Train a model and then test it."""

    def __init__(
        self,
        net,
        ds_tr,
        ds_te,
        ds_vl=None,
        metrics_tr=[esp.metrics.TypingCrossEntropy()],
        metrics_te=[esp.metrics.TypingCrossEntropy()],
        optimizer=lambda net: torch.optim.Adam(net.parameters(), 1e-2),
        normalize=esp.data.normalize.NotNormalize,
        n_epochs=100,
        record_interval=1,
        device=torch.device("cpu"),
        scheduler=None,
    ):

        # bookkeeping
        self.device = device
        self.net = net
        self.ds_tr = ds_tr
        self.ds_te = ds_te
        self.ds_vl = ds_vl
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.metrics_tr = metrics_tr
        self.metrics_te = metrics_te
        self.normalize = normalize
        self.record_interval = record_interval
        self.scheduler = scheduler

    def __str__(self):
        _str = ""
        _str += "# model"
        _str += "\n"
        _str += str(self.net)
        _str += "\n"
        if hasattr(self.net, "noise_model"):
            _str += "# noise model"
            _str += "\n"
            _str += str(self.net.noise_model)
            _str += "\n"
        _str += "# optimizer"
        _str += "\n"
        _str += str(self.optimizer)
        _str += "\n"
        _str += "# n_epochs"
        _str += "\n"
        _str += str(self.n_epochs)
        _str += "\n"
        return _str

    def run(self):
        """Run train and test."""
        train = Train(
            net=self.net,
            data=self.ds_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            metrics=self.metrics_tr,
            normalize=self.normalize,
            device=self.device,
            record_interval=self.record_interval,
            scheduler=self.scheduler,
        )

        train.train()

        self.states = train.states

        test = Test(
            net=self.net,
            data=self.ds_te,
            metrics=self.metrics_te,
            states=self.states,
            normalize=self.normalize,
            device=self.device,
        )

        test.test()

        self.ref_g_test = test.ref_g

        self.results_te = test.results

        test = Test(
            net=self.net,
            data=self.ds_tr,
            metrics=self.metrics_te,
            states=self.states,
            normalize=self.normalize,
            device=self.device,
        )

        test.test()
        self.ref_g_training = test.ref_g

        self.results_tr = test.results

        if self.ds_vl is not None:

            test = Test(
                net=self.net,
                data=self.ds_vl,
                metrics=self.metrics_te,
                states=self.states,
                normalize=self.normalize,
                device=self.device,
            )

            test.test()
            self.ref_g_validation = test.ref_g

            self.results_vl = test.results

            return {
                "test": self.results_te,
                "train": self.results_tr,
                "validate": self.results_vl,
            }

        return {"test": self.results_te, "train": self.results_tr}
