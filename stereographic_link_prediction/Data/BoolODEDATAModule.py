import requests
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .GeneDatasets import (
    TrainLinkPredictionDataset,
    ValLinkPredictionDataset,
)
from .GeneDataModule import split_edges


class GSD(pl.LightningDataModule):
    url = "https://raw.githubusercontent.com/Murali-group/Beeline/master/inputs/example/GSD"
    expression_data = "ExpressionData.csv"
    ref_networks = "refNetwork.csv"

    def __init__(
        self,
        data_dir: str,
        *,
        batch_size: int = 64,
        batch_size_val: int = 2 * 4096,
        num_workers: int = 20,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers

    def prepare_data(self):
        for f in [self.expression_data, self.ref_networks]:
            r = requests.get(f"{self.url}/{f}", allow_redirects=False)
            open(os.path.join(self.data_dir, f), "wb").write(r.content)

    def setup(self, stage: str):
        data = pd.read_csv(
            os.path.join(self.data_dir, self.expression_data), index_col=0
        )
        reg = pd.read_csv(os.path.join(self.data_dir, self.ref_networks))

        node_features = data.to_numpy()
        node_features = (node_features - node_features.mean()) / node_features.std()

        genes = data.index

        edges = []
        regulation_modes = ["+", "-"]
        for r, g, m in zip(reg["Gene1"], reg["Gene2"], reg["Type"]):
            ind1 = genes.get_loc(r)
            ind2 = genes.get_loc(g)

            edges.append([ind1, ind2, regulation_modes.index(m)])

        self.edges = np.array(edges)
        edges_ind = self.edges[:, 2] == 0
        edges0 = self.edges[edges_ind, :]
        edges1 = self.edges[np.logical_not(edges_ind), :]

        self.input_dim = node_features.shape[1]
        self.output_dim = len(regulation_modes)

        train_edges0, val_edges0, test_edges0 = split_edges(edges0)
        train_edges1, val_edges1, test_edges1 = split_edges(edges1)
        train_edges = np.concatenate((train_edges0, train_edges1), axis=0)
        val_edges = np.concatenate((val_edges0, val_edges1), axis=0)
        test_edges = np.concatenate((test_edges0, test_edges1), axis=0)

        train_nodes, val_nodes, test_nodes = split_edges(
            np.arange(node_features.shape[0])
        )

        if stage == "fit" or stage is None:
            self.train_dataset = TrainLinkPredictionDataset(
                train_edges, train_nodes, node_features
            )

            self.valid_dataset = ValLinkPredictionDataset(val_edges, node_features)

            print(f"Train set length: {len(self.train_dataset)}")
            print(f"Validation set length: {len(self.valid_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = ValLinkPredictionDataset(test_edges, node_features)

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
