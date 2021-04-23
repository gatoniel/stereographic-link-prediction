from typing import Union
import torch
import pytorch_lightning as pl
import geoopt

from geoopt import (
    Scaled,
    StereographicProductManifold,
    PoincareBall,
    SphereProjection,
    Stereographic,
    ManifoldParameter,
)

from ..Data.Distances import DistancesDataModule


class MDS(pl.LightningModule):
    def __init__(
        self,
        datamodule: Union[pl.LightningDataModule, None] = None,
        *,
        hyperbolic_dim: int = 2,
        spherical_dim: int = 0,
        euclidean_dim: int = 0,
        learning_rate: float = 0.5,
        use_scheduler: bool = False,
        learnable_curvature: bool = False,
        learnable_scale: bool = False,
        data_dir: str = "./data",
        replica: int = 1,
        batch_size: int = 1024,
        num_workers: int = 20,
    ):
        super().__init__()
        if datamodule is None:
            self.datamodule = DistancesDataModule(
                data_dir=data_dir,
                replica=replica,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            self.datamodule.prepare_data()
            self.datamodule.setup("fit")
        else:
            self.datamodule = datamodule
        self.num_points = len(self.datamodule)

        manifolds = [
            (
                PoincareBall(learnable=learnable_curvature),
                hyperbolic_dim,
            ),
            (
                SphereProjection(learnable=learnable_curvature),
                spherical_dim,
            ),
            (
                Stereographic(k=0, learnable=learnable_curvature),
                euclidean_dim,
            ),
        ]
        manifolds = tuple((m for m in manifolds if m[1] > 0))
        self.manifold = StereographicProductManifold(*manifolds)
        self.latent_dim = sum([m[1] for m in manifolds])
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        self.points = ManifoldParameter(
            self.manifold.random(self.num_points, self.latent_dim),
            manifold=self.manifold,
        )
        self.register_parameter(
            "log_scale", torch.nn.Parameter(torch.tensor(0.0))
        )

    @property
    def scale(self):
        return self.log_scale.exp()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LinkPredictionModule")

        parser.add_argument("--hyperbolic_dim", type=int, default=2)
        parser.add_argument("--spherical_dim", type=int, default=0)
        parser.add_argument("--euclidean_dim", type=int, default=0)

        parser.add_argument("--learning_rate", type=float, default=1)
        parser.add_argument("--use_scheduler", action="store_true")

        parser.add_argument("--learnable_curvature", action="store_true")
        parser.add_argument("--learnable_scale", action="store_true")
        return parent_parser

    def forward(self, x):
        return self.points

    def calc_distances(self, indexes):
        return self.manifold.dist(
            self.points[indexes[:, 0], :], self.points[indexes[:, 1], :]
        )

    def training_step(self, batch, batch_idx):
        d_ij, indexes = batch

        d_hat_ij = self.calc_distances(indexes)
        # d_ij_ = d_ij.detach()
        # d_hat_ij_ = d_hat_ij.detach()
        # a = (d_ij_ * d_hat_ij_).sum() / d_hat_ij_.pow(2).sum()
        # a = (d_ij * d_hat_ij).sum() / d_hat_ij.pow(2).sum()

        # loss = torch.mean((d_ij - self.log_scale.exp() * d_hat_ij) ** 2)
        loss = torch.mean((d_ij - d_hat_ij) ** 2)
        # loss = torch.mean((d_ij - a * d_hat_ij) ** 2)

        self.log("train_loss", loss)
        self.log("scale", self.scale.detach())
        self.log("min_d_hat", d_hat_ij.min())
        self.log("max_d_hat", d_hat_ij.max())
        for i, mf in enumerate(self.manifold.manifolds):
            self.log(f"curvature{i}", mf.k)

        return loss

    def training_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        tensorboard.add_image(
            "embedding",
            self.datamodule.plot(self.points, self.manifold.manifolds[0]),
            self.trainer.current_epoch,
        )

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def configure_optimizers(self):
        optimizer = geoopt.optim.RiemannianAdam(
            self.parameters(), lr=self.learning_rate
        )
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_loss",
            }
        else:
            return optimizer
