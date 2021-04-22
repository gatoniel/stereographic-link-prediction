import io
import matplotlib.pyplot as plt
import torch
import torchvision
import geoopt

from PIL import Image


def detach(l):
    return [x.detach().cpu() for x in l]


def plot_connections(ax, z1_, z2_, y_, manifold):
    z1, z2, y = detach([z1_, z2_, y_])
    ax.scatter(z1[:, 0], z1[:, 1], marker="o")
    ax.scatter(z2[:, 0], z2[:, 1], marker=">")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    t = torch.linspace(0, 1, 50).unsqueeze(-1).repeat(1, 2).type_as(z1_)
    for i in range(z1.shape[0]):
        if y[i] >= 0:
            geodesic = manifold.geodesic(t, z1_[i, :], z2_[i, :]).detach().cpu()
            ax.plot(geodesic[:, 0], geodesic[:, 1], color=colors[y[i]], alpha=0.5)


def fig_poincare_ball(manifold):
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    ax.set_aspect(aspect=1)

    ax.add_patch(
        plt.Circle((0, 0), manifold.radius.item(), edgecolor="black", facecolor="none")
    )

    return fig, ax


def connections_to_png(z1, z2, y, manifold):
    if isinstance(manifold, geoopt.PoincareBall):
        fig, ax = fig_poincare_ball(manifold)
        plot_connections(ax, z1, z2, y, manifold)
        return fig_to_img(fig)


def plot_hyperplanes(ax, points_, direction_, manifold):
    direction_ = direction_ / direction_.norm(dim=-1, keepdim=True)
    points, direction = detach([points_, direction_])
    ax.scatter(points[:, 0], points[:, 1], marker="o")
    ax.quiver(points[:, 0], points[:, 1], direction[:, 0], direction[:, 1])

    orthogonal = torch.ones_like(direction_)
    orthogonal[:, 0] = -direction_[:, 1] / direction_[:, 0]
    # orthogonal = direction_

    t = torch.linspace(-2, 2, 500).unsqueeze(-1).repeat(1, 2).type_as(points_)
    for i in range(points.shape[0]):
        geodesic = (
            manifold.geodesic_unit(t, points_[i, :], orthogonal[i, :]).detach().cpu()
        )
        ax.plot(geodesic[:, 0], geodesic[:, 1], color="black", alpha=0.5)


def hyperplanes_to_png(points, direction, manifold):
    if isinstance(manifold, geoopt.PoincareBall):
        fig, ax = fig_poincare_ball(manifold)
        plot_hyperplanes(ax, points, direction, manifold)
        return fig_to_img(fig)


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return torchvision.transforms.ToTensor()(img)
