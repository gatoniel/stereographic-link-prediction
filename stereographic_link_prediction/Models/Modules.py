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
from torchmetrics import (
    MeanSquaredError,
    RetrievalMAP,
    RetrievalPrecision,
    RetrievalMRR,
)

from .Networks import (
    EncoderWrapped,
    DecoderDist2Plane,
    Discriminator2DifferentDist2Plane,
    # Discriminator2SameDist2Plane,
)

from ..utils.plot import connections_to_png, hyperplanes_to_png


class LinkPredictionModule(pl.LightningModule):
    def __init__(
        self,
        hparams,
        datamodule: pl.LightningDataModule,
        *,
        reconstruction_criterion=None,
        alpha: int = 1,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.datamodule = datamodule
        self.input_dim = datamodule.input_dim
        self.output_dim = datamodule.output_dim
        # self.output_dim = datamodule.output_dim + 1

        manifolds = [
            (PoincareBall(learnable=True), hparams.hyperbolic_dim),
            (SphereProjection(learnable=True), hparams.spherical_dim),
            (Stereographic(k=0, learnable=True), hparams.euclidean_dim),
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
        self.discriminator = Discriminator2DifferentDist2Plane(
            self.output_dim,
            self.latent_dim,
            hparams.discriminator_hidden_dim,
            self.manifold,
        )

        self.reconstruction_loss = nn.MSELoss()
        self.discrimination_loss = nn.CrossEntropyLoss()
        # self.register_buffer(
        #     "inv_y", torch.tensor((self.output_dim - 1)), persistent=False
        # )

        self.metrics = nn.ModuleDict(
            {
                "train_mse": MeanSquaredError(),
                "val_mse": MeanSquaredError(),
                "test_mse": MeanSquaredError(),
            }
        )
        for i in range(self.output_dim):
            # for i in range(self.output_dim - 1):
            self.metrics.update(
                {
                    # f"val_{i}_MAP": MeanAbsoluteError(),
                    # f"test_{i}_MAP": MeanAbsoluteError(),
                    # f"val_{i}_Precision": MeanAbsoluteError(),
                    # f"test_{i}_Precision": MeanAbsoluteError(),
                    f"train_{i}_MAP": RetrievalMAP(),
                    f"val_{i}_MAP": RetrievalMAP(),
                    f"test_{i}_MAP": RetrievalMAP(),
                    f"train_{i}_Precision": RetrievalPrecision(),
                    f"val_{i}_Precision": RetrievalPrecision(),
                    f"test_{i}_Precision": RetrievalPrecision(),
                    f"train_{i}_MRR": RetrievalMRR(),
                    f"val_{i}_MRR": RetrievalMRR(),
                    f"test_{i}_MRR": RetrievalMRR(),
                }
            )

        self.step = {"train": 0, "val": 0, "test": 0}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LinkPredictionModule")

        parser.add_argument("--encoder_layers", type=int, default=1)
        parser.add_argument("--encoder_hidden_dim", type=int, default=10)

        parser.add_argument("--decoder_layers", type=int, default=1)
        parser.add_argument("--decoder_hidden_dim", type=int, default=10)

        parser.add_argument("--discriminator_layers", type=int, default=1)
        parser.add_argument("--discriminator_hidden_dim", type=int, default=10)

        parser.add_argument("--hyperbolic_dim", type=int, default=2)
        parser.add_argument("--spherical_dim", type=int, default=2)
        parser.add_argument("--euclidean_dim", type=int, default=0)

        parser.add_argument("--learning_rate", type=float, default=1e-2)
        return parent_parser

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x1, x2, x3, y, i1 = batch
        x = torch.cat((x1, x2, x3), axis=0)

        # mean, std = self(x)
        # z = self.manifold.wrapped_normal(mean, std, mean.shape)
        z = self(x)
        z1, z2, z3 = torch.chunk(z, 3, dim=0)
        x3_ = self.decoder(z3)

        rec_loss = self.reconstruction_loss(x3, x3_)

        pred = self.discriminator(z1, z2)
        # randomly predict that there is no edge between nodes
        # pred_inv = self.discriminator(z1, z3)
        # pred_inv = self.discriminator(z2, z1)

        discr_loss = self.discrimination_loss(pred, y)
        # discr_inv_loss = self.discrimination_loss(
        #     pred_inv, self.inv_y.repeat(pred_inv.shape[0])
        # )
        loss = rec_loss + discr_loss  # + discr_inv_loss

        self.log("train_loss", loss)
        self.log("train_rec", rec_loss)
        self.log("train_discr", discr_loss)
        # self.log("train_discr_inv", discr_inv_loss)

        return {
            "loss": loss,
            "i1": i1,
            "pred1": pred,
            "y": y,
            "x": x3,
            "x_": x3_,
            "z1": z1,
            "z2": z2,
        }

    def training_step_end(self, outputs):
        self.shared_val_step_end(outputs, "train")
        return outputs["loss"]

    def training_epoch_end(self, outputs):
        self.shared_epoch_end([], "train")

    def shared_val_step(self, batch, batch_idx):
        x1, x2, _, y, i1 = batch
        x = torch.cat((x1, x2), axis=0)
        # mean, _ = self(x)
        mean = self(x)
        x_ = self.decoder(mean)
        z1, z2 = torch.chunk(mean, 2, dim=0)

        pred1 = F.softmax(self.discriminator(z1, z2), dim=-1)

        return {
            "i1": i1,
            "pred1": pred1,
            "y": y,
            "x": x,
            "x_": x_,
            "z1": z1,
            "z2": z2,
        }

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch, batch_idx)

    def shared_val_step_end(self, outputs, name_):
        name = f"{name_}_mse"
        self.metrics[name](outputs["x"], outputs["x_"])
        for i in range(self.output_dim):
            # for i in range(self.output_dim - 1):
            target = outputs["y"] == i
            for t in ["MAP", "Precision", "MRR"]:
                name = f"{name_}_{i}_{t}"
                self.metrics[name](
                    # outputs["pred1"][:, i],
                    # outputs["pred1"][:, i] ** 2
                    outputs["i1"],
                    outputs["pred1"][:, i],
                    target,
                )
        return outputs

    def shared_epoch_end(self, outputs, name_):
        name = f"{name_}_mse"
        self.log(name, self.metrics[name].compute())

        for i in range(self.output_dim):
            # for i in range(self.output_dim - 1):
            for t in ["MAP", "Precision", "MRR"]:
                name = f"{name_}_{i}_{t}"
                self.log(name, self.metrics[name].compute())

        if len(outputs) > 0:
            tensorboard = self.logger.experiment
            mf = self.manifold
            num_edges = 200
            outputs = outputs[0]
            z1 = outputs["z1"][:num_edges, :]
            z2 = outputs["z2"][:num_edges, :]
            y = outputs["y"][:num_edges]
            tensorboard.add_image(
                f"{name_}_connections",
                connections_to_png(
                    mf.take_submanifold_value(z1, 0),
                    mf.take_submanifold_value(z2, 0),
                    y,
                    mf.manifolds[0],
                ),
                self.step[name_],
            )
            for i, layer in enumerate(
                [
                    self.discriminator.dist2plane1,
                    self.discriminator.dist2plane2,
                ]
            ):
                tensorboard.add_image(
                    f"{name_}_hyperplanes_{i}",
                    hyperplanes_to_png(
                        mf.take_submanifold_value(layer.point, 0).squeeze(),
                        mf.take_submanifold_value(
                            layer.direction, 0
                        ).squeeze(),
                        mf.manifolds[0],
                    ),
                    self.step[name_],
                )
            self.step[name_] += 1

    def validation_step_end(self, outputs):
        return self.shared_val_step_end(outputs, "val")

    def test_step_end(self, outputs):
        return self.shared_val_step_end(outputs, "test")

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return geoopt.optim.RiemannianAdam(
            self.parameters(), lr=self.learning_rate
        )
