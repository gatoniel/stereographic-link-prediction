from argparse import ArgumentParser
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers

from stereographic_link_prediction.Data.GeneDataModule import (
    BacillusSubtilisSwarming,
)
from stereographic_link_prediction.Data.BoolODEDATAModule import GSD
from stereographic_link_prediction.Models.Modules import LinkPredictionModule


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LinkPredictionModule.add_model_specific_args(parser)
    hparams = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger("./logs")

    # datamodule = BacillusSubtilisSwarming("./data", batch_size=4096)
    datamodule = GSD("./data", batch_size=128)
    datamodule.prepare_data()
    datamodule.setup("fit")
    module = LinkPredictionModule(hparams, datamodule)

    trainer = pl.Trainer(
        gpus=1,
        precision=64,
        # logging
        logger=tb_logger,
        log_every_n_steps=1,
        # deterministic=True,
        # profiler="simple",
        # reload_dataloaders_every_epoch=True,
        check_val_every_n_epoch=100,
        max_epochs=10000,
        # gradient_clip_val=0.5,
        # terminate_on_nan=True,
        track_grad_norm=2,
        # overfit_batches=10,
        # num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
