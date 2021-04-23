import os
import torch
import geoopt
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from math import floor

from ..utils import plot


class DistancesDataset(Dataset):
    def __init__(self, distances):
        super().__init__()

        assert distances.shape[0] == distances.shape[1]
        self.distances = distances
        self.indexes = torch.combinations(
            torch.arange(self.distances.shape[0]), r=2
        ).numpy()

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        ind = self.indexes[idx, :]
        return self.distances[ind[0], ind[1]], ind


class DistancesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        replica: int = 1,
        *,
        batch_size: int = 64,
        num_workers: int = 20,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.replica = replica

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = None

    def __len__(self):
        return self.length

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        distances = pd.read_csv(
            os.path.join(self.data_dir, f"Rep{self.replica}_distance.csv"),
            index_col=0,
        )
        spacetime = pd.read_csv(
            os.path.join(self.data_dir, f"Rep{self.replica}_spacetime.csv"),
            index_col="sample #",
        ).T
        index = [i.replace("_", "") for i in distances.index]
        spacetime = spacetime[index].T
        self.time = spacetime["Timepoint after lag phase"].to_numpy()
        self.space = spacetime["Position in swarm"].to_numpy()

        distances = distances.T.to_numpy()
        distances = np.delete(distances, 69, 0)
        distances = np.delete(distances, 69, 1)
        self.time = np.delete(self.time, 69, 0)
        self.space = np.delete(self.space, 69, 0)
        self.length = distances.shape[0]

        if stage == "fit" or stage is None:
            self.train_dataset = DistancesDataset(distances)

        difference = 0.1
        un = np.unique(self.time)
        self.times = np.empty(len(un) - (np.diff(un) < difference).sum())
        self.times[0] = un[0]
        self.times[1:] = un[1:][np.diff(un) > difference]

        self.inds_min = []
        self.inds_max = []
        self.inds_med = []
        self.inds_time = []
        for i in range(len(self.times)):
            inds = np.abs(self.time - self.times[i]) < difference
            tmp_inds = np.arange(len(self.time))[inds]
            tmp_space = self.space[inds]

            sort_max = np.argmax(tmp_space)
            sort_min = np.argmin(tmp_space)
            sort_med = np.argsort(tmp_space)[len(tmp_space) // 2]

            self.inds_min.append(tmp_inds[sort_min])
            self.inds_max.append(tmp_inds[sort_max])
            self.inds_med.append(tmp_inds[sort_med])

            sort_inds = np.argsort(tmp_space)
            tmp_inds2 = tmp_inds[sort_inds]
            self.inds_time.append(tmp_inds2)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def plot(self, points, manifold):
        points = points.detach().cpu()
        fig, axes = plot.fig_poincare_ball(manifold, (1, 2), (10, 5))
        titles = ["time", "space"]
        vals = [self.time, self.space]
        for i in range(len(axes)):
            ax = axes[i]
            vmin = vals[i].min()
            vmax = vals[i].max()
            sc = ax.scatter(
                points[:, 0], points[:, 1], c=vals[i], vmin=vmin, vmax=vmax
            )
            fig.colorbar(sc, ax=ax)
            ax.set_title(titles[i])

        for i in range(len(self.times)):
            axes[1].plot(
                points[self.inds_time[i], 0],
                points[self.inds_time[i], 1],
                alpha=0.4,
                label=f"{self.times[i]:1.1f}",
            )

        lists = [self.inds_min, self.inds_max, self.inds_med]
        labels = [
            "Minimal space trajectory",
            "Max space trajectory",
            "Median space trajectory",
        ]
        for i in range(len(lists)):
            axes[0].plot(
                points[lists[i], 0],
                points[lists[i], 1],
                alpha=0.4,
                label=labels[i],
            )
        # for i in range(2):
        #     axes[i].legend()

        return plot.fig_to_img(fig)


class RandomDistancesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int = 64,
        num_workers: int = 20,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = None

    def __len__(self):
        return self.length

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        mf = geoopt.PoincareBall()
        dim = 2
        num_per_dest = 30
        num_dest = 3
        num = num_dest * num_per_dest
        destinations = mf.random(num_dest, dim)
        t = torch.linspace(0, 1, num_per_dest)
        t = t.unsqueeze(1).repeat(1, num_dest).unsqueeze(2)
        o = mf.origin(dim).unsqueeze(0).repeat(3, 1)
        points = mf.geodesic(t, o, destinations).reshape(-1, dim)
        noise = mf.random(*points.shape, std=0.05)
        self.points = mf.mobius_add(points, noise)

        indexes = torch.combinations(torch.arange(num), r=2)
        distances = mf.dist(
            self.points[indexes[:, 0], :], self.points[indexes[:, 1], :]
        )
        tri = np.zeros((num, num))
        tri[indexes[:, 0], indexes[:, 1]] = distances

        self.length = tri.shape[0]

        if stage == "fit" or stage is None:
            self.train_dataset = DistancesDataset(tri)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def plot(self, points, manifold):
        points = points.detach().cpu()
        fig, axes = plot.fig_poincare_ball(manifold, (1, 2), (8, 4))
        titles = ["real points", "estimated points"]
        vals = [self.points.detach().cpu(), points]
        for i in range(len(axes)):
            ax = axes[i]
            ax.scatter(
                vals[i][:, 0],
                vals[i][:, 1],
                c=np.arange(points.shape[0]),
                vmin=0,
                vmax=points.shape[0] - 1,
            )
            ax.set_title(titles[i])

        return plot.fig_to_img(fig)
