import torch
import geoopt
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from stereographic_link_prediction.Data.Distances import (
    DistancesDataModule,
    RandomDistancesDataModule,
)

if __name__ == "__main__":
    datamodule = DistancesDataModule("./data", batch_size=None)
    datamodule.setup("fit")
    # for i in tqdm(range(len(datamodule))):
    #     fig, ax = plt.subplots(1, 1)
    #     ax.scatter(datamodule.time[i], datamodule.space[i], color="red")
    #     times = []
    #     spaces = []
    #     vals = []
    #     for j in range(len(datamodule)):
    #         if i == j:  # or j in [69]:
    #             continue
    #         if i < j:
    #             vals.append(datamodule.train_dataset.distances[i, j])
    #         else:
    #             vals.append(datamodule.train_dataset.distances[j, i])
    #         times.append(datamodule.time[j])
    #         spaces.append(datamodule.space[j])
    #     sc = ax.scatter(
    #         times, spaces, c=vals, vmin=min(vals), vmax=max(vals), cmap="jet"
    #     )
    #     fig.colorbar(sc, ax=ax)

    #     fig.savefig(f"/mnt/d/distances/{i}.png")
    #     plt.close(fig)

    fig = datamodule.plot(
        torch.tensor(datamodule.flat_mds), geoopt.PoincareBall(), True
    )
    fig.savefig(f"/mnt/d/distances/mds.png")

    indexes = torch.combinations(
        torch.arange(datamodule.flat_mds.shape[0]), r=2
    ).numpy()
    flat_mds0 = datamodule.flat_mds[indexes[:, 0]]
    flat_mds1 = datamodule.flat_mds[indexes[:, 1]]
    dist = np.sqrt(
        (flat_mds0[:, 0] - flat_mds1[:, 0]) ** 2
        + (flat_mds0[:, 1] - flat_mds1[:, 1]) ** 2
    )
    dist_true = datamodule.train_dataset.distances[
        indexes[:, 0], indexes[:, 1]
    ]
    loss = np.mean((dist_true - dist) ** 2)
    print("LOSS: ", loss)
