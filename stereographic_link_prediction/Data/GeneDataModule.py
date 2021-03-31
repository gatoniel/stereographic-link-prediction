import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from math import floor

from .GeneDatasets import (
    TrainLinkPredictionDataset,
    ValLinkPredictionDataset,
)


def read_data(path: str, replica: int = 1) -> pd.DataFrame:
    counts_file = os.path.join(path, f"logcounts_rep{replica}.csv")
    counts = pd.read_csv(counts_file, index_col=0).T
    regulations = pd.read_csv(os.path.join(path, "regulations.csv"))
    translation = pd.read_excel(
        os.path.join(path, "Matching_geneNamesSubtiwiki.xlsx")
    ).applymap(lambda x: x.replace("'", ""))
    return counts, regulations, translation


def simplify_protein(p):
    p = p[0].lower() + p[1:]
    return p


def split_edges(edges: np.array, train: float = 0.8, val: float = 0.1):
    length = edges.shape[0]
    train = floor(train * length)
    val = train + floor(val * length)

    indices = np.random.permutation(edges.shape[0])
    t_inds = indices[:train]
    v_inds = indices[train:val]
    test_inds = indices[val:]

    return edges[t_inds], edges[v_inds], edges[test_inds]


class BacillusSubtilisSwarming(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        replica: int = 1,
        *,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.replica = replica

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        counts, regulations, translation = read_data(self.data_dir, self.replica)
        node_features = counts.T.to_numpy()
        node_features = (node_features - node_features.mean()) / node_features.std()

        genes = counts.columns
        translation.index = translation["Gene name in subtiwiki"]

        regulations["regulator2"] = [
            simplify_protein(r) for r in regulations["regulator"]
        ]

        edges = []
        edge_nodes = []
        regulation_modes = ["activation", "repression"]
        for r, g, m in zip(
            regulations["regulator2"], regulations["gene"], regulations["mode"]
        ):
            if m not in regulation_modes:
                continue
            # we try as there might happen errors in the translation step
            try:
                trans_r_ind = translation.index.get_loc(r)
                trans_g_ind = translation.index.get_loc(g)

                trans_r = translation["Our gene identifier"][trans_r_ind]
                trans_g = translation["Our gene identifier"][trans_g_ind]

                ind1 = genes.get_loc(trans_r)
                ind2 = genes.get_loc(trans_g)
                edges.append([ind1, ind2, regulation_modes.index(m)])
                edge_nodes.append(ind1)
                edge_nodes.append(ind2)
            except KeyError:
                pass
        edge_nodes = np.array(edge_nodes)
        other_nodes = np.setdiff1d(np.arange(node_features.shape[0]), edge_nodes)
        self.edges = np.array(edges)

        self.input_dim = node_features.shape[1]
        self.output_dim = len(regulation_modes) + 1

        train_edges, val_edges, test_edges = split_edges(self.edges)
        _, val_nodes, test_nodes = split_edges(edge_nodes)
        train_nodes = other_nodes

        if stage == "fit" or stage is None:
            self.train_dataset = TrainLinkPredictionDataset(
                train_edges, train_nodes, node_features
            )

            self.valid_dataset = ValLinkPredictionDataset(val_edges, node_features)

        if stage == "test" or stage is None:
            self.test_dataset = ValLinkPredictionDataset(test_edges, node_features)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
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
