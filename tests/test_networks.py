import torch
import numpy as np
import geoopt

from stereographic_link_prediction.Models import Networks


def test_EncoderWrapped():
    ball = geoopt.PoincareBall()
    encoder = Networks.EncoderWrapped(10, 2, 5, ball)

    x = torch.randn(20, 10)
    mu, sigma = encoder(x)

    ball.assert_check_point_on_manifold(mu)
    assert torch.all(sigma > 0)
    assert mu.shape == sigma.shape
    assert mu.shape == torch.Size([20, 2])


def test_DecoderWrapped():
    ball = geoopt.PoincareBall()
    decoder = Networks.DecoderWrapped(10, 2, 5, ball)

    x = ball.random(20, 2)
    y = decoder(x)

    assert y.shape == torch.Size([20, 10])


def test_LayerDist2Plane():
    ball = geoopt.PoincareBall()
    layer = Networks.LayerDist2Plane(10, 2, ball)

    x = ball.random(20, 2)
    y = layer(x)

    assert y.shape == torch.Size([20, 10])
    for i in range(x.shape[0]):
        assert torch.allclose(y[i, :], layer(x[i, :]))


def test_adam_LayerDist2Plane():
    ball = geoopt.PoincareBall()
    latent_dim = 3
    layer = Networks.LayerDist2Plane(1, latent_dim, ball).double()

    ideal_point = torch.tensor([0.5, 0.0, 0.0])  # .double()
    ideal_bias = torch.zeros(1, 1)  # .double()

    def closure():
        optim.zero_grad()
        x = ball.random(100, latent_dim)  # .double()
        y = (
            ball.dist2plane(x, ideal_point, -ideal_point, signed=True) + ideal_bias
        ).transpose(1, 0)
        loss = torch.mean((layer(x) - y) ** 2)
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam(layer.parameters(), lr=1e-2)

    for _ in range(10000):
        optim.step(closure)

    distance1 = ball.dist2plane(
        ideal_point[0, ...], layer.point.detach(), layer.direction.detach()
    )
    distance2 = ball.dist2plane(layer.point.detach(), ideal_point, -ideal_point)
    distances = np.array([distance1.item(), distance2.item()])
    np.testing.assert_allclose(distances, np.zeros(2), atol=1e-2, rtol=1e-2)
    # np.testing.assert_allclose(
    #     ideal_weight.cpu(), layer.weight.detach().cpu(), atol=1e-5, rtol=1e-5
    # )
    np.testing.assert_allclose(
        ideal_bias.cpu(), layer.bias.detach().cpu(), atol=1e-5, rtol=1e-5
    )


def test_DecoderDist2Plane():
    ball = geoopt.PoincareBall()
    decoder = Networks.DecoderDist2Plane(10, 2, 5, ball)

    x = ball.random(20, 2)
    y = decoder(x)

    assert y.shape == torch.Size([20, 10])


def test_Discriminator2SameDist2Plane():
    ball = geoopt.PoincareBall()
    decoder = Networks.Discriminator2SameDist2Plane(10, 2, 5, ball)

    x = ball.random(20, 2)
    y = ball.random(20, 2)
    y = decoder(x, y)

    assert y.shape == torch.Size([20, 10])


def test_Discriminator2DifferentDist2Plane():
    ball = geoopt.PoincareBall()
    decoder = Networks.Discriminator2DifferentDist2Plane(10, 2, 5, ball)

    x = ball.random(20, 2)
    y = ball.random(20, 2)
    y = decoder(x, y)

    assert y.shape == torch.Size([20, 10])
