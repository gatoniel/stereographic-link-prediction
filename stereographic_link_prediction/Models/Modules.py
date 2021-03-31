import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import geoopt

from geoopt import (
    StereographicProductManifold,
    PoincareBall,
    SphereProjection,
    Stereographic,
)
from torchmetrics import RetrievalMAP, RetrievalPrecision, MetricCollection

from .Networks import (
    EncoderWrapped,
    DecoderDist2Plane,
    Discriminator2SameDist2Plane,
)


class LinkPredictionModule(pl.LightningModule):
    def __init__(
        self,
        hparams,
        datamodule: pl.LightningDataModule,
        *,
        lr: float = 0.0001,
        reconstruction_criterion=None,
        alpha: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.input_dim = datamodule.input_dim
        self.output_dim = datamodule.output_dim

        manifolds = [
            (PoincareBall(), hparams.hyperbolic_dim),
            (SphereProjection(), hparams.spherical_dim),
            (Stereographic(k=0), hparams.euclidean_dim),
        ]
        manifolds = tuple((m for m in manifolds if m[1] > 0))
        self.manifold = StereographicProductManifold(*manifolds)
        self.latent_dim = sum([m[1] for m in manifolds])
        self.learning_rate = hparams.learning_rate

        self.encoder = EncoderWrapped(
            self.input_dim,
            self.latent_dim,
            hparams.encoder_hidden_dim,
            self.manifold,
            depths=hparams.encoder_layers,
        )
        self.decoder = DecoderDist2Plane(
            self.input_dim,
            self.latent_dim,
            hparams.decoder_hidden_dim,
            self.manifold,
            depths=hparams.decoder_layers,
        )
        self.discriminator = Discriminator2SameDist2Plane(
            self.output_dim,
            self.latent_dim,
            hparams.discriminator_hidden_dim,
            self.manifold,
        )

        self.reconstruction_loss = nn.MSELoss()
        self.discrimination_loss = nn.CrossEntropyLoss()
        self.inv_y = torch.tensor((self.output_dim - 1))

        for i in range(self.output_dim - 1):
            self.__dict__[f"val_{i}"] = MetricCollection(
                [RetrievalMAP(), RetrievalPrecision(k=20)]
            )
            self.__dict__[f"test_{i}"] = MetricCollection(
                [RetrievalMAP(), RetrievalPrecision(k=20)]
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LinkPredictionModule")

        parser.add_argument("--encoder_layers", type=int, default=4)
        parser.add_argument("--encoder_hidden_dim", type=int, default=100)

        parser.add_argument("--decoder_layers", type=int, default=4)
        parser.add_argument("--decoder_hidden_dim", type=int, default=100)

        parser.add_argument("--discriminator_layers", type=int, default=1)
        parser.add_argument("--discriminator_hidden_dim", type=int, default=50)

        parser.add_argument("--hyperbolic_dim", type=int, default=2)
        parser.add_argument("--spherical_dim", type=int, default=2)
        parser.add_argument("--euclidean_dim", type=int, default=2)

        parser.add_argument("--learning_rate", type=int, default=1e-3)
        return parent_parser

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x1, x2, x3, y = batch
        x = torch.cat((x1, x2, x3), axis=0)

        mean, std = self(x)
        z = self.manifold.wrapped_normal(mean, std, mean.shape)
        z1, z2, z3 = torch.chunk(z, 3, dim=0)
        x3_ = self.decoder(z3)

        rec_loss = self.reconstruction_loss(x3, x3_)

        pred = self.discriminator(z1, z2)
        pred_inv = self.discriminator(z2, z1)

        discr_loss = self.discrimination_loss(pred, y)
        discr_inv_loss = self.discrimination_loss(
            pred_inv, self.inv_y.repeat(pred_inv.shape[0])
        )
        self.log("train_rec", rec_loss)
        self.log("train_discr", discr_loss)
        self.log("train_discr_inv", discr_inv_loss)

        return rec_loss + discr_loss + discr_inv_loss

    def shared_val_step(self, batch, batch_idx):
        x1, x2, y, i1 = batch
        x = torch.cat((x1, x2), axis=0)
        mean, _ = self(x)
        z1, z2 = torch.chunk(mean, 2, dim=0)

        pred1 = F.softmax(self.discriminator(z1, z2), dim=-1)

        return {"i1": i1, "pred1": pred1, "y": y}

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch, batch_idx)

    def shared_val_step_end(self, outputs, name_):
        for i in range(self.output_dim - 1):
            name = f"{name_}_{i}"
            self.__dict__[name](
                outputs["i1"], outputs["pred1"][:, i], outputs["y"] == i
            )
            self.log(name, self.__dict__[name].compute())

    def validation_step_end(self, outputs):
        self.shared_val_step_end(outputs, "val")

    def test_step_end(self, outputs):
        self.shared_val_step_end(outputs, "test")

    def configure_optimizers(self):
        return geoopt.optim.RiemannianAdam(self.parameters(), lr=self.learning_rate)
