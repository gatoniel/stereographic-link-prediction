from argparse import ArgumentParser
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from stereographic_link_prediction.Data.Distances import (
    DistancesDataModule,
    RandomDistancesDataModule,
)
from stereographic_link_prediction.Models.MDS import MDS


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MDS.add_model_specific_args(parser)
    parser = DistancesDataModule.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument_group("program specific")
    parser.add_argument("--tb_path", type=str, default="./MDS_logs")
    parser.add_argument("--chpt_path", type=str, default="./chpt")
    parser.add_argument(
        "--chpt_name", type=str, default="MDS-{epoch:02d}-{train_loss:.2f}"
    )

    args = parser.parse_args()

    dict_args = vars(args)

    datamodule = DistancesDataModule(**dict_args)
    datamodule.prepare_data()
    datamodule.setup("fit")

    module = MDS(datamodule, **dict_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.chpt_path,
        filename=args.chpt_name,
        mode="min",
    )
    tb_logger = pl_loggers.TensorBoardLogger(args.tb_path)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=tb_logger
    )

    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)
