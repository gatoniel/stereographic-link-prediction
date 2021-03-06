from argparse import ArgumentParser
import torch.multiprocessing
from types import SimpleNamespace
import pytorch_lightning as pl

from stereographic_link_prediction.Data.GeneDataModule import (
    BacillusSubtilisSwarming,
)
from stereographic_link_prediction.Models.Modules import LinkPredictionModule

pl.seed_everything(42)
torch.multiprocessing.set_sharing_strategy("file_system")


def test_fast_dev_run():
    parser = ArgumentParser()
    parser = LinkPredictionModule.add_model_specific_args(parser)
    hparams = SimpleNamespace(
        encoder_layers=4,
        encoder_hidden_dim=100,
        decoder_layers=4,
        decoder_hidden_dim=100,
        discriminator_layers=1,
        discriminator_hidden_dim=50,
        hyperbolic_dim=2,
        spherical_dim=2,
        euclidean_dim=2,
        learning_rate=1e-3,
    )

    datamodule = BacillusSubtilisSwarming("./data")
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    module = LinkPredictionModule(hparams, datamodule)

    trainer = pl.Trainer(fast_dev_run=True, precision=32)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)


def test_overfit():
    hparams = SimpleNamespace(
        encoder_layers=4,
        encoder_hidden_dim=100,
        decoder_layers=4,
        decoder_hidden_dim=100,
        discriminator_layers=1,
        discriminator_hidden_dim=50,
        hyperbolic_dim=2,
        spherical_dim=2,
        euclidean_dim=2,
        learning_rate=1e-3,
    )

    datamodule = BacillusSubtilisSwarming("./data")
    datamodule.prepare_data()
    datamodule.setup("fit")
    module = LinkPredictionModule(hparams, datamodule)

    trainer = pl.Trainer(overfit_batches=10, precision=32)
    trainer.fit(module, datamodule=datamodule)
