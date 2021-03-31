import numpy as np
from torch.utils.data import Dataset


class TrainLinkPredictionDataset(Dataset):
    def __init__(
        self,
        edges,
        nodes,
        node_features,
    ):
        super().__init__()

        assert edges.shape[1] == 3
        assert edges.dtype == int
        assert edges.ndim == node_features.ndim == 2

        self.edges = edges
        self.nodes = np.random.permutation(nodes)
        self.node_features = node_features.astype(np.float32)

        self.edges_len = edges.shape[0]
        self.nodes_len = nodes.shape[0]

    def __len__(self):
        return max(self.edges_len, self.nodes_len)

    def __getitem__(self, idx):
        myedge = self.edges[idx % self.edges_len, :]

        start_feature = self.node_features[myedge[0], :]
        end_feature = self.node_features[myedge[1], :]
        link_feature = myedge[2]

        additional_node = self.nodes[idx % self.nodes_len]
        additional_feature = self.node_features[additional_node, :]

        return (
            start_feature,
            end_feature,
            additional_feature,
            link_feature,
        )


class ValLinkPredictionDataset(Dataset):
    def __init__(
        self,
        edges,
        node_features,
    ):
        super().__init__()

        assert edges.shape[1] == 3
        assert edges.dtype == int
        assert edges.ndim == node_features.ndim == 2

        self.edges = edges
        self.node_features = node_features.astype(np.float32)

        total_num_nodes = self.node_features.shape[0]
        connection_matrix = -np.ones((total_num_nodes, total_num_nodes), dtype=int)
        for i in range(self.edges.shape[0]):
            connection_matrix[self.edges[i, 0], self.edges[i, 1]] = self.edges[i, 2]

        empty_rows = connection_matrix == -1
        empty_inds = empty_rows.all(axis=1)

        x = np.arange(total_num_nodes)
        x, y = np.meshgrid(x, x, indexing="ij")
        connection_matrix = np.stack((x, y, connection_matrix), axis=-1)

        connection_matrix = connection_matrix[np.logical_not(empty_inds), ...]
        assert connection_matrix.shape[0] == len(np.unique(self.edges[:, 0]))

        connection_matrix = connection_matrix.reshape(-1, 3)
        diagonal_inds = connection_matrix[:, 0] == connection_matrix[:, 1]
        self.connection_matrix = connection_matrix[np.logical_not(diagonal_inds), :]

    def __len__(self):
        return self.connection_matrix.shape[0]

    def __getitem__(self, idx):
        myedge = self.connection_matrix[idx, :]

        start_feature = self.node_features[myedge[0], :]
        end_feature = self.node_features[myedge[1], :]
        link_feature = myedge[2]

        return (
            start_feature,
            end_feature,
            link_feature,
            myedge[0],
        )
