import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import geoopt

from torch.nn.parameter import Parameter
from geoopt import ManifoldParameter


def Linears(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depths: int = 1,
):
    linears = [
        nn.Linear(in_features=input_dim, out_features=hidden_dim),
        nn.ReLU(),
    ]
    for i in range(depths):
        linears.append(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )
        linears.append(nn.ReLU())
    linears.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))
    return nn.Sequential(*linears)


class EncoderWrapped(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        manifold,
        *,
        depths: int = 1,
        eta: float = 1e-5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.eta = eta
        self.linears = Linears(input_dim, 2 * latent_dim, hidden_dim, depths)
        self.manifold = manifold

    def forward(self, x):
        x = self.linears(x).view(-1, self.latent_dim, 2)
        mu = self.manifold.expmap0(x[..., 0])
        sigma = F.softplus(x[..., 1]) + self.eta
        return mu, sigma


class DecoderWrapped(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        manifold,
        depths: int = 1,
    ):
        super().__init__()
        self.linears = Linears(latent_dim, output_dim, hidden_dim, depths)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)
        return self.linears(x)


class LayerDist2Plane(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        manifold: geoopt.Manifold,
    ):
        super().__init__()
        self.manifold = manifold
        self.point = ManifoldParameter(
            manifold.random(output_dim, 1, latent_dim), manifold=manifold
        )
        sphere = geoopt.Sphere()
        self.direction = geoopt.ManifoldParameter(
            sphere.random(output_dim, 1, latent_dim), manifold=sphere
        )

        self.manifold = manifold
        self.bias = Parameter(torch.empty(output_dim, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.bias)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.manifold.dist2plane(
                x, self.point, self.direction, signed=True
            )
            + self.bias
        ).transpose(1, 0)


class DecoderDist2Plane(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        manifold,
        depths: int = 1,
    ):
        super().__init__()
        self.manifold = manifold
        self.dist2plane = LayerDist2Plane(hidden_dim, latent_dim, manifold)
        self.linears = Linears(hidden_dim, output_dim, hidden_dim, depths)

    def forward(self, x):
        return self.linears(self.dist2plane(x))


class Discriminator2SameDist2Plane(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        manifold,
    ):
        super().__init__()
        self.manifold = manifold
        self.dist2plane = LayerDist2Plane(hidden_dim, latent_dim, manifold)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, output_dim)

    def forward(self, x, y):
        x = self.dist2plane(x)
        y = self.dist2plane(y)
        return self.bilinear(x, y)


class Discriminator2DifferentDist2Plane(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        manifold,
    ):
        super().__init__()
        self.manifold = manifold
        self.dist2plane1 = LayerDist2Plane(hidden_dim, latent_dim, manifold)
        self.dist2plane2 = LayerDist2Plane(hidden_dim, latent_dim, manifold)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, output_dim)

    def forward(self, x, y):
        x = self.dist2plane1(x)
        y = self.dist2plane2(y)
        return self.bilinear(x, y)
