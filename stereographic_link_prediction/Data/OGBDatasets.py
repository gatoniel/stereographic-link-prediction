import numpy as np
from torch.utils.data import Dataset


class ValNegLinkPredictionDataset(Dataset):
    def __init__(self, source, target, target_neg, node_features):
        super().__init__()

        self.source = source
        self.target = target
        self.target_neg = target_neg
        self.node_features = node_features.astype(np.float32)

        self.shape = (len(self.source), self.target_neg.shape[1] + 1)

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __getitem__(self, idx):
        source_index, target_index = np.unravel_index(idx, self.shape)

        start_feature = self.node_features[self.source[source_index], :]
        try:
            end_feature = self.node_features[
                self.target_neg[source_index, target_index], :
            ]
            link_feature = -1
        except IndexError:
            end_feature = self.node_features[self.target[source_index], :]
            link_feature = 0

        return (
            start_feature,
            end_feature,
            end_feature,
            link_feature,
            self.source[source_index],
        )
