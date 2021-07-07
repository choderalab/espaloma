import numpy as onp
from jax import grad


def jax_play_nice_with_scipy(jax_loss_fxn):
    """Defining functions that can talk to scipy optimizers"""

    def fun(params):
        return float(jax_loss_fxn(params))  # make sure return type is a plain old float)

    g = grad(jax_loss_fxn)

    def jac(params):
        return onp.array(g(params), dtype=onp.float64)  # make sure return type is a float64 array

    return fun, jac