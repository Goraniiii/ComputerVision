import numpy as np
from sklearn.metrics import pairwise_distances


class KNearestNeighbor:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.distance = None
        self.metric = None
        self.sorted_indices = None

    def train(self, X, y):
        """in KNN, train is just saving the train data"""
        self.X_train = X
        self.y_train = y

    def compute_distance(self, X, metric='euclidean'):
        self.distance = pairwise_distances(X, self.X_train, metric=metric)

        self.sorted_indices = np.argsort(self.distance, axis=1)

        # print(f"DONE: computing distance for {metric} metric")


    def predict(self, k = 3):
        """
        Compute label prediction
        k
        return (M,)
        """

        if self.distance is None:
            raise Exception('Need to train first')
        if self.sorted_indices is None:
            raise Exception('Need to train first')

        nearest_idx = self.sorted_indices[:, :k]
        neighbor_labels = self.y_train[nearest_idx]

        y_pred = np.array([np.bincount(row).argmax() for row in neighbor_labels])
        return y_pred


    # def compute_l1_distance(self, X_test):
    #     """
    #     L1: Manhattan distance
    #     return (M, N) matrix
    #     M: number of test data
    #     N: number of training data
    #     """
    #     distances = np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
    #     return distances
    #
    #
    # def compute_l2_distance(self, X_test):
    #     """
    #     L2: Manhattan distance
    #     return (M, N) matrix
    #     M: number of test data
    #     N: number of training data
    #     """
    #
    #     distances = np.sqrt(np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
    #     return distances