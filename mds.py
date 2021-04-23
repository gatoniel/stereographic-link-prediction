from argparse import ArgumentParser
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers

from stereographic_link_prediction.Data.Distances import (
    DistancesDataModule,
    RandomDistancesDataModule,
)
from stereographic_link_prediction.Models.MDS import MDS


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MDS.add_model_specific_args(parser)

    parser.add_argument(
        "--datamodule", choices=["toy", "data"], default="data"
    )

    hparams = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger("./MDS_logs")

    if hparams.datamodule == "data":
        datamodule = DistancesDataModule("./data")
    else:
        datamodule = RandomDistancesDataModule()
    datamodule.prepare_data()
    datamodule.setup("fit")
    module = MDS(hparams, datamodule)

    trainer = pl.Trainer(
        gpus=1,
        precision=64,
        # logging
        logger=tb_logger,
        log_every_n_steps=2,
        # deterministic=True,
        profiler="simple",
        # reload_dataloaders_every_epoch=True,
        # max_epochs=10,
        # gradient_clip_val=0.5,
        # terminate_on_nan=True,
        # track_grad_norm=2,
        # overfit_batches=10,
        # num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=datamodule)
