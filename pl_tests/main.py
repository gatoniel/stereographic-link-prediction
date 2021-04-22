from argparse import ArgumentParser
import torch.multiprocessing
import pytorch_lightning as pl

from stereographic_link_prediction.Data.GeneDataModule import (
    BacillusSubtilisSwarming,
)
from stereographic_link_prediction.Models.Modules import LinkPredictionModule

pl.seed_everything(42)
# torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LinkPredictionModule.add_model_specific_args(parser)
    hparams = parser.parse_args()

    datamodule = BacillusSubtilisSwarming("./data")
    datamodule.prepare_data()
    datamodule.setup("fit")
    module = LinkPredictionModule(hparams, datamodule)

    # trainer = pl.Trainer(auto_scale_batch_size="power", precision=32)
    # trainer.tune(module, datamodule=datamodule)

    trainer = pl.Trainer(auto_lr_find=True, precision=32)
    trainer.tune(module, datamodule=datamodule)
