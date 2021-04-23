from pytorch_lightning.utilities.cli import LightningCLI

from stereographic_link_prediction.Models.MDS import MDS


cli = LightningCLI(MDS)
