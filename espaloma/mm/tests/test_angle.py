import pytest
import torch
import numpy as np
import numpy.testing as npt 

def test_angle_random_vectors():
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
            loc=torch.zeros(3, ),
            scale=torch.ones(3, ))

    left = distribution.sample()
    right = distribution.sample()

    cos_ref = (left * right).sum(dim=-1) \
            / (torch.norm(left) * torch.norm(right))

    cos_hat = torch.cos(esp.mm.geometry._angle(left, right))

    npt.assert_almost_equal(
            cos_ref.numpy(),
            cos_hat.numpy())

def test_angle_random_points():
    import espaloma as esp

    distribution = torch.distributions.normal.Normal(
            loc=torch.zeros(5, 3),
            scale=torch.ones(5, 3))

    x0 = distribution.sample()
    x1 = distribution.sample()
    x2 = distribution.sample()

    left = x1 - x0
    right = x1 - x2

    cos_ref = (left * right).sum(dim=-1) \
            / (torch.norm(left, dim=-1) * torch.norm(right, dim=-1))

    cos_hat = torch.cos(esp.angle(x0, x1, x2))

    npt.assert_almost_equal(
            cos_ref.numpy(),
            cos_hat.numpy())


def test_zero():
    import espaloma as esp

    x0 = torch.zeros(5, 3)
    
    npt.assert_almost_equal(
            esp.angle(x0, x0, x0).numpy(),
            0.0)








