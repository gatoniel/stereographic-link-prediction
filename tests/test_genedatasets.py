import numpy as np

from stereographic_link_prediction.Data import GeneDatasets


def test_ValLinPredictionDataset():
    edges = np.array(
        [
            [0, 1, 0],
            [0, 2, 0],
            [2, 0, 1],
        ]
    )
    connection_matrix = np.stack(
        (
            np.array(
                [
                    [0, 0],
                    [2, 2],
                ]
            ),
            np.array(
                [
                    [1, 2],
                    [0, 1],
                ]
            ),
            np.array(
                [
                    [0, 0],
                    [1, -1],
                ]
            ),
        ),
        axis=-1,
    ).reshape(-1, 3)

    node_features = np.random.randn(3, 2)

    dataset = GeneDatasets.ValLinkPredictionDataset(edges, node_features)
    np.testing.assert_array_equal(dataset.connection_matrix, connection_matrix)

    batch = dataset[0]
    np.testing.assert_array_equal(batch[0], node_features[0, :])
    np.testing.assert_array_equal(batch[1], node_features[1, :])
    assert batch[2] == batch[3] == 0

    assert len(dataset) == 4


def test_TrainLinkPredictionDataset():
    edges = np.array(
        [
            [0, 1, 0],
            [0, 3, 0],
            [1, 3, 0],
        ]
    )
    node_features = np.random.randn(4, 2)
    nodes = np.array([2])

    dataset = GeneDatasets.TrainLinkPredictionDataset(
        edges, nodes, node_features
    )
    assert len(dataset) == 3

    batch = dataset[0]
    np.testing.assert_array_equal(batch[0], node_features[0])
    np.testing.assert_array_equal(batch[1], node_features[1])
    np.testing.assert_array_equal(batch[2], node_features[2])
    assert batch[3] == 0
