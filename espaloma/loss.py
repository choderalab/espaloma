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
def mse(y, y_hat):
    return torch.nn.functional.mse_loss(y, y_hat)

def rmse(y, y_hat):
    return torch.sqrt(torch.nn.functional.mse_loss(y, y_hat))

def cross_entropy(y, y_hat, reduction='mean'):
    loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    return loss_fn(y_hat, y) # prediction first, logit

def accuracy(y, y_hat):
    # check if this is logit
    if y.dim() == 2 and y.shape[-1] > 1:
        y = y.argmax(dim=-1)

    return torch.div(
            torch.sum(
                1.0 * torch.equal(y, y_hat)),
            y.shape[0])

def r2(y, y_hat):
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Loss(torch.nn.modules.loss._Loss):
    """ Base function for loss.

    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(Loss, self).__init__(size_average, reduce, reduction)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class GraphLoss(Loss):
    """ Loss between nodes attributes of graph or graphs.

    """
    def __init__(self, base_loss, between, *args, **kwargs):
        super(GraphLoss, self).__init__(*args, **kwargs)
        
        # between could be tuple of two strings or two functions
        assert len(between) == 2

        self.between = (
                self._translation(between[0]),
                self._translation(between[1]))

        self.base_loss = base_loss

    @staticmethod
    def _translation(string):
        return {
            'nn_typing': lambda g: g.ndata['nn_typing'],
            'legacy_typing': lambda g: g.ndata['legacy_typing']
        }[string]

    
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
        return self.base_loss.forward(
                input=input_fn(g_input),
                target=target_fn(g_target))



