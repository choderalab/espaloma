import numpy as np
import numpy.testing as npt
import pytest
import torch


def test_distance():
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
        loc=torch.zeros(5, 3), scale=torch.ones(5, 3)
    )

    x0 = distribution.sample()
    x1 = distribution.sample()

    npt.assert_almost_equal(
        esp.distance(x0, x1).numpy(),
        torch.sqrt((x0 - x1).pow(2).sum(dim=-1)).numpy(),
        decimal=3,
    )

    npt.assert_almost_equal(esp.distance(x0, x0).numpy(), 0.0)
