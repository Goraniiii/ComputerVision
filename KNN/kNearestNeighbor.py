import numpy as np

class KNearestNeighbor:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """in KNN, train is just saving the data"""
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k = 3, distance_metric = 'l2'):
        """
        Compute label prediction
        X_test : (M, D) test dta
        k
        distance_metric : l1 or l2

        return (M,)
        """

        if distance_metric == 'l1':
            distances = self.compute_l1_distance(X_test)
        elif distance_metric == 'l2':
            distances = self.compute_l2_distance(X_test)
        else:
            raise NotImplementedError

        M = X_test.shape[0]
        y_pred = np.zeros(M, dtype=self.y_train.dtype)

        for i in range(M):
            k_closest_indices = np.argsort(distances[i, :])[:k]

            k_nearest_labels = self.y_train[k_closest_indices]

            counts = np.bincount(k_nearest_labels)
            y_pred[i] = np.argmax(counts)

        return y_pred


    def compute_l1_distance(self, X_test):
        """
        L1: Manhattan distance
        return (M, N) matrix
        M: number of test data
        N: number of training data
        """
        M = X_test.shape[0]
        N = self.X_train.shape[0]
        distances = np.zeros((M, N))

        for i in range(M):
            distances[i, :] = np.sum(np.abs(self.X_train[i] - X_test[i]), axis=1)

        return distances

    def compute_l2_distance(self, X_test):
        """
        L2: Manhattan distance
        return (M, N) matrix
        M: number of test data
        N: number of training data
        """
        X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)
        X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True)
        dot_product = X_test.dot(X_test_sq)

        distances_sq = X_test_sq - 2 * dot_product + X_train_sq

        return np.maximum(0, distances_sq)