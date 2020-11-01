""" Metrics to evaluate and train models.

"""
import abc

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def center(metric, weight=1.0, dim=1):
    def _centered(input, target, metric=metric, weight=weight, dim=dim):
        # center input
        input = input - input.mean(dim=dim, keepdim=True)

        # center target
        target = target - target.mean(dim=dim, keepdim=True)

        return weight * metric(input, target)

    return _centered


def std(metric, weight=1.0, dim=1):
    def _std(input, target, metric=metric, weight=weight, dim=dim):
        return weight * metric(input, target).std(dim=dim).sum()

    return _std

def weighted(metric, weight, reduction="mean"):
    def _weighted(
            input, target, metric=metric, weight=weight, reduction=reduction
        ):
        _loss = metric(input, target)
        for _ in range(_loss.dims()-1):
            weight = weight.unsqueeze(-1)
        return getattr(torch, reduction)(weight)
    return _weighted

def weighted_with_key(metric, key="weight", reduction="mean"):
    def _weighted(input, target, metric=metric, key=key, reduction=reduction):
        weight = target.nodes["g"].data[key].flatten()
        _loss = metric(input, target)
        for _ in range(_loss.dims()-1):
            weight = weight.unsqueeze(-1)
        return getattr(torch, reduction)(weight)
    return _weighted

def bootstrap(metric, n_samples=1000, ci=0.95):
    def _bootstrap(input, target, metric=metric, n_samples=n_samples, ci=0.95):
        original = metric(input=input, target=target)

        idxs_all = np.arange(input.shape[0])
        results = []
        for _ in range(n_samples):
            idxs = np.random.choice(idxs_all, len(idxs_all), replace=True,)

            _metric = (
                metric(input=input[idxs], target=target[idxs])
                .detach()
                .cpu()
                .numpy()
                .item()
            )

            results.append(_metric,)

        results = np.array(results)

        low = np.percentile(results, 100.0 * 0.5 * (1 - ci))
        high = np.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)

        return original.detach().cpu().numpy().item(), low, high

    return _bootstrap


def latex_format_ci(original, low, high):
    return "%.4f_{%.4f}^{%.4f}" % (original, low, high)


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mse(input, target):
    return torch.nn.functional.mse_loss(target, input)


def mape(input, target):
    return ((input - target).abs() / target.abs()).mean()


def rmse(input, target):
    return torch.sqrt(torch.nn.functional.mse_loss(target, input))


def mae_of_log(input, target):
    return torch.nn.L1Loss()(torch.log(input), torch.log(target))


def cross_entropy(input, target, reduction="mean"):
    loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    return loss_fn(input=input, target=target)  # prediction first, logit


def r2(input, target):
    target = target.flatten()
    input = input.flatten()
    ss_tot = (target - target.mean()).pow(2).sum()
    ss_res = (input - target).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)


def accuracy(input, target):
    # check if this is logit
    if input.dim() == 2 and input.shape[-1] > 1:
        input = input.argmax(dim=-1)

    return torch.div((input == target).sum().double(), target.shape[0])


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Metric(torch.nn.modules.loss._Loss):
    """ Base function for loss.

    """

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(Metric, self).__init__(size_average, reduce, reduction)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class GraphMetric(Metric):
    """ Loss between nodes attributes of graph or graphs.

    """

    def __init__(self, base_metric, between, level="n1", *args, **kwargs):
        super(GraphMetric, self).__init__(*args, **kwargs)

        # between could be tuple of two strings or two functions
        assert len(between) == 2

        self.between = (
            self._translation(between[0], level),
            self._translation(between[1], level),
        )

        self.base_metric = base_metric

        # get name
        if hasattr(base_metric, "__name__"):
            base_name = base_metric.__name__
        else:
            base_name = base_metric.__class__.__name__

        self.__name__ = "%s_between_%s_and_%s_on_%s" % (
            base_name,
            between[0],
            between[1],
            level,
        )

    @staticmethod
    def _translation(string, level):
        return lambda g: g.nodes[level].data[string]

    def forward(self, g_input, g_target=None):
        """ Forward function of loss.

        """
        # allow loss within self
        if g_target is None:
            g_target = g_input

        # get input and output transform function
        input_fn, target_fn = self.between

        # compute loss using base loss
        # NOTE:
        # use keyward argument here since torch is bad with the order with args
        return self.base_metric(
            input=input_fn(g_input), target=target_fn(g_target)
        )


class GraphDerivativeMetric(Metric):
    """ Loss between nodes attributes of graph or graphs.

    """

    def __init__(
        self,
        base_metric,
        between,
        weight=1.0,
        level="n1",
        d="xyz",
        d_level="n1",
        *args,
        **kwargs
    ):
        super(GraphDerivativeMetric, self).__init__(*args, **kwargs)

        # between could be tuple of two strings or two functions
        assert len(between) == 2

        self.between = (
            self._translation(between[0], level),
            self._translation(between[1], level),
        )

        self.d = self._translation(d, d_level)

        self.base_metric = base_metric
        self.weight = weight
        # get name
        if hasattr(base_metric, "__name__"):
            base_name = base_metric.__name__
        else:
            base_name = base_metric.__class__.__name__

        self.__name__ = "%s_between_d_%s_d_%s_and_d_%s_d_%s_on_%s" % (
            base_name,
            between[0],
            d,
            between[1],
            d,
            level,
        )

    @staticmethod
    def _translation(string, level):
        return lambda g: g.nodes[level].data[string]

    def forward(self, g_input, g_target=None):
        """ Forward function of loss.

        """
        # allow loss within self
        if g_target is None:
            g_target = g_input

        # get input and output transform function
        input_fn, target_fn = self.between

        # calculate the derivatives of input
        input_prime = torch.autograd.grad(
            input_fn(g_input).sum(),
            self.d(g_input),
            create_graph=True,
            retain_graph=True,
        )[0]

        target_prime = torch.autograd.grad(
            target_fn(g_target).sum(),
            self.d(g_target),
            create_graph=True,
            retain_graph=True,
        )[0]

        # compute loss using base loss
        # NOTE:
        # use keyward argument here since torch is bad with the order with args
        return self.weight * self.base_metric(
            input=input_prime, target=target_prime,
        )


class GraphHalfDerivativeMetric(Metric):
    """ Loss between nodes attributes of graph or graphs.

    """

    def __init__(
        self,
        base_metric,
        input_level="g",
        input_name="u",
        target_prime_level="n1",
        target_prime_name="u_ref_prime",
        d="xyz",
        d_level="n1",
        weight=1.0,
        *args,
        **kwargs
    ):
        super(GraphHalfDerivativeMetric, self).__init__(*args, **kwargs)

        # define query functions
        self.d = self._translation(d, d_level)
        self.input_fn = self._translation(input_name, input_level)
        self.target_prime_fn = self._translation(
            target_prime_name, target_prime_level
        )

        self.base_metric = base_metric
        self.weight = weight
        # get name
        if hasattr(base_metric, "__name__"):
            base_name = base_metric.__name__
        else:
            base_name = base_metric.__class__.__name__

        self.__name__ = "%s_between_%s_d_%s_on_%s_and_%s_on_%s" % (
            base_name,
            input_name,
            d,
            input_level,
            target_prime_name,
            target_prime_level,
        )

    @staticmethod
    def _translation(string, level):
        return lambda g: g.nodes[level].data[string]

    def forward(self, g_input, g_target=None):
        """ Forward function of loss.

        """
        # allow loss within self
        if g_target is None:
            g_target = g_input

        # calculate the derivatives of input
        input_prime = torch.autograd.grad(
            self.input_fn(g_input).sum(),
            self.d(g_input),
            create_graph=True,
            retain_graph=True,
        )[0]

        target_prime = self.target_prime_fn(g_target)

        # compute loss using base loss
        # NOTE:
        # use keyward argument here since torch is bad with the order with args
        return self.weight * self.base_metric(
            input=input_prime, target=target_prime,
        )


# =============================================================================
# PRESETS
# =============================================================================


class TypingCrossEntropy(GraphMetric):
    def __init__(self):
        super(TypingCrossEntropy, self).__init__(
            base_metric=torch.nn.CrossEntropyLoss(),
            between=["nn_typing", "legacy_typing"],
        )

        self.__name__ = "TypingCrossEntropy"


class TypingAccuracy(GraphMetric):
    def __init__(self):
        super(TypingAccuracy, self).__init__(
            base_metric=accuracy, between=["nn_typing", "legacy_typing"]
        )

        self.__name__ = "TypingAccuracy"


class BondKMSE(GraphMetric):
    def __init__(self):
        super(BondKMSE, self).__init__(
            between=["k_ref", "k"], level="n2", base_metric=torch.nn.MSELoss()
        )

        self.__name__ = "BondKMSE"


class BondKRMSE(GraphMetric):
    def __init__(self):
        super(BondKRMSE, self).__init__(
            between=["k_ref", "k"], level="n2", base_metric=rmse
        )

        self.__name__ = "BondKRMSE"
