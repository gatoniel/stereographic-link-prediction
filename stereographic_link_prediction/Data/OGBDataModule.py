import os
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset

from .GeneDatasets import TrainLinkPredictionDataset
from .OGBDatasets import ValNegLinkPredictionDataset


class Citation2(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 4096,
        batch_size_val: int = 2 * 4096,
        num_workers: int = 20,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers

    def prepare_data(self):
        LinkPropPredDataset("ogbl-citation2", root=self.data_dir)

    def setup(self, stage: str):
        dataset = LinkPropPredDataset("ogbl-citation2", root=self.data_dir)
        node_features = dataset[0]["node_feat"]
        node_features = (
            node_features - node_features.mean()
        ) / node_features.std()

        split_edge = dataset.get_edge_split()

        train_edges = np.stack(
            (
                split_edge["train"]["source_node"],
                split_edge["train"]["target_node"],
                np.zeros_like(
                    split_edge["train"]["source_node"],
                ),
            ),
            axis=-1,
        )
        train_nodes = np.setdiff1d(
            np.arange(node_features.shape[0]),
            np.concatenate(
                (
                    split_edge["train"]["source_node"],
                    split_edge["train"]["target_node"],
                )
            ),
        )

        self.input_dim = node_features.shape[1]
        self.output_dim = 2

        if stage == "fit" or stage is None:
            self.train_dataset = TrainLinkPredictionDataset(
                train_edges, train_nodes, node_features
            )

            self.valid_dataset = ValNegLinkPredictionDataset(
                split_edge["valid"]["source_node"],
                split_edge["valid"]["target_node"],
                split_edge["valid"]["target_node_neg"],
                node_features,
            )

            print(f"Train set length: {len(self.train_dataset)}")
            print(f"Validation set length: {len(self.valid_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = ValNegLinkPredictionDataset(
                split_edge["test"]["source_node"],
                split_edge["test"]["target_node"],
                split_edge["test"]["target_node_neg"],
                node_features,
            )

            print(f"Test set length: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
