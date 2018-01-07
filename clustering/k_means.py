"""
K-means clustering for Local Histogram-Based Color Image Clustering
with custom similarity function.
"""

# Author: Tomasz Bochenski

import numpy as np


class LocalHistogramKMeans(object):
    """
    K-means clustering for Local Histogram-Based Color Image Clustering
    with custom similarity function.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.

    init : {'k-means++' or 'random'}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean clustering in a
        smart way to speed up convergence.

        'random' : choose k observations (rows) at random from data for
        the initial centroids.

    n_init : int, default: 10
        Number of times the k-mean algorithm will be run with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms
        of sum_similarity.

    max_iter : int, default: 300
        Maximum number of iterations of k-means algorithm for a single run.

    tol : float, default: 1e-4
        Relative tolerance with regards to sum_similarity to declare convergence.

    random_state_seed : int, default: None
        The seed used by the random number generator.

    Attributes
    ----------
    cluster_centers_ : array
        Coordinates of cluster centers.

    labels_ : array
        Labels of each point.

    similarities_ : array
        Similarities for each cluster.

    sum_similarities : float
        Mean similarities for all clusters.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, random_state_seed=None):
        self.__n_clusters = n_clusters
        self.__init = init
        self.__n_init = n_init
        self.__max_iter = max_iter
        self.__tol = tol
        self.random_state_seed = random_state_seed

        self.random_state = np.random.RandomState(self.random_state_seed)

        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(self.__n_clusters, dtype=float)
        self.sum_similarities_ = None

        self.__prev_similarities = None

        self.__best_labels = None
        self.__best_cluster_centers = None
        self.__best_similarities = None
        self.__best_sum_similarities = None

    def __local_reset(self):
        """Resets local state of the object."""
        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(self.__n_clusters, dtype=float)
        self.sum_similarities_ = None

        self.__prev_similarities = None

    def __global_reset(self):
        """Resets global state of the object."""
        self.random_state = np.random.RandomState(self.random_state_seed)
        self.__best_labels = None
        self.__best_cluster_centers = None
        self.__best_similarities = None
        self.__best_sum_similarities = None

    @staticmethod
    def __calculate_hue_distances(h_q, h_t):
        """Calculates distances between hues."""
        return ((np.cos(np.absolute(h_q - h_t) * 2.0 * np.pi / 256.0)) ** 2) / 2.0

    @staticmethod
    def __calculate_saturation_distances(s_q, s_t):
        """Calculates distances between saturations."""
        return np.absolute(s_q - s_t) / 256.0

    @staticmethod
    def __calculate_value_distances(v_q, v_t):
        """Calculate distances between values."""
        return np.absolute(v_q - v_t) / 256.0

    @staticmethod
    def __calculate_similarities(data, cluster_centers):
        """Calculates similarity of images."""
        parameter_a = 2.5
        parameter_b = 0.5
        parameter_c = 0.0

        hue_data = data[range(data.shape[0]), :, :, 0]
        saturation_data = data[range(data.shape[0]), :, :, 1]
        value_data = data[range(data.shape[0]), :, :, 2]
        hue_centers = cluster_centers[range(cluster_centers.shape[0]), :, 0]
        saturation_centers = cluster_centers[range(cluster_centers.shape[0]), :, 1]
        value_centers = cluster_centers[range(cluster_centers.shape[0]), :, 2]

        dh = LocalHistogramKMeans.__calculate_hue_distances(hue_data, hue_centers)
        ds = LocalHistogramKMeans.__calculate_saturation_distances(saturation_data, saturation_centers)
        dv = LocalHistogramKMeans.__calculate_value_distances(value_data, value_centers)

        block_similarities = 1.0 / (1.0 + parameter_a * dh + parameter_b * ds + parameter_c * dv)
        return np.mean(block_similarities, axis=2)

    def __random_init(self, data):
        """Initializes cluster centers randomly."""
        idx = self.random_state.permutation(data.shape[0])[:self.__n_clusters]
        self.cluster_centers_ = np.copy(data[idx])

    def __kmeans_pp_init(self, data):
        """Initializes cluster centers using 'k-means++' algorithm."""
        self.cluster_centers_ = data[np.random.choice(range(data.shape[0]), 1), :]
        ext_data = data[:, np.newaxis, :]
        while self.cluster_centers_.shape[0] < self.__n_clusters:
            similarities = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)
            self.labels_ = np.argmax(similarities, axis=1)
            max_similarities = np.max(similarities, axis=1)
            non_similarities = 1.0 - max_similarities
            non_similarities_sum = np.sum(non_similarities)
            prob = non_similarities / non_similarities_sum
            self.cluster_centers_ = np.vstack([self.cluster_centers_, data[np.random.choice(
                range(data.shape[0]), 1, p=prob), :]])

    def __assign_labels(self, data):
        """Assigns labels for data."""
        ext_data = data[:, np.newaxis, :]
        similarities = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)
        self.labels_ = np.argmax(similarities, axis=1)
        clusters_not_empty = np.isin(range(self.__n_clusters), self.labels_)
        while not np.all(clusters_not_empty):
            empty_clusters = np.logical_not(clusters_not_empty)
            self.cluster_centers_[empty_clusters] = data[self.random_state.random_integers(
                0, len(data) - 1, np.sum(empty_clusters))]
            similarities = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)
            self.labels_ = np.argmax(similarities, axis=1)
            clusters_not_empty = np.isin(range(self.__n_clusters), self.labels_)

    def __update_centers(self, data):
        """Computes new claster centers."""
        for i in xrange(self.__n_clusters):
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            self.cluster_centers_[i] = np.mean(points, axis=0)

    def __update_similarities(self, data):
        """Calculates new similarities."""
        for i in xrange(self.__n_clusters):
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            ext_points = points[:, np.newaxis, :]
            cluster = self.cluster_centers_[i][np.newaxis, :]
            similarities = LocalHistogramKMeans.__calculate_similarities(ext_points, cluster)
            self.similarities_[i] = np.mean(similarities)
        self.sum_similarities_ = np.mean(self.similarities_)

    def __is_progress(self):
        """Checks whether to break the algorithm or continue."""
        if self.__prev_similarities is None:
            return True
        prev_similarities_sorted = np.sort(self.__prev_similarities)
        similarities_sorted = np.sort(self.similarities_)
        return np.any(np.abs(prev_similarities_sorted - similarities_sorted) > self.__tol)

    def __update_best_solution(self):
        #print  self.sum_similarities_
        """Updates the best solution if a new solution is better than previous ones."""
        #print self.__best_sum_similarities
        if self.__best_sum_similarities is None or self.__best_sum_similarities < self.sum_similarities_:
            self.__best_labels = np.copy(self.labels_)
            self.__best_cluster_centers = np.copy(self.cluster_centers_)
            self.__best_similarities = np.copy(self.similarities_)
            self.__best_sum_similarities = self.sum_similarities_

    def __load_best_solution(self):
        """Loads the best solution calculated by the algorithm in 'n_init' runs."""
        self.labels_ = np.copy(self.__best_labels)
        self.cluster_centers_ = np.copy(self.__best_cluster_centers)
        self.similarities_ = np.copy(self.__best_similarities)
        self.sum_similarities_ = self.__best_sum_similarities

    def __call__(self, data):
        """Performs k-means algorithm for Local Histogram-Based Color Image Clustering task."""
        self.__global_reset()
        for i in xrange(self.__n_init):
            self.__local_reset()

            if self.__init == 'k-means++':
                self.__kmeans_pp_init(data)
            elif self.__init == 'random':
                self.__random_init(data)
            else:
                raise ValueError('n_init: Inappropriate initialization method')

            for iteration in xrange(self.__max_iter):
                self.__assign_labels(data)
                self.__update_centers(data)
                self.__update_similarities(data)
                if not self.__is_progress():
                    break
                self.__prev_similarities = np.copy(self.similarities_)
            self.__update_best_solution()

        self.__load_best_solution()
