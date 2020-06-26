""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import abc

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mse(input, target):
    return torch.nn.functional.mse_loss(target, input)


def rmse(input, target):
    return torch.sqrt(torch.nn.functional.mse_loss(target, input))


def cross_entropy(input, target, reduction="mean"):
    loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    return loss_fn(input=input, target=target)  # prediction first, logit


def r2(target, input):
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

        # get base name
        if hasattr(base_metric, "__init__"):
            base_name = base_metric.__class__.__name__
        else:
            base_name = base_metric.__name__

        self.__name__ = "%s between %s and %s" % (base_name, between[0], between[1])

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
        return self.base_metric(input=input_fn(g_input), target=target_fn(g_target))


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
