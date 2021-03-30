from types import SimpleNamespace
from stereographic_link_prediction.Models.Modules import LinkPredictionModule
from stereographic_link_prediction.Data.GeneDataModule import (
    BacillusSubtilisSwarming,
)


def test_latentdim():
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
    datamodule.setup("train")
    module = LinkPredictionModule(hparams, datamodule)

    assert module.latent_dim == 6
