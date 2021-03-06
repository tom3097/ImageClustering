"""
K-means clustering for Global Histogram-Based Color Image Clustering
with euclidean distance function.
"""

import numpy as np


class GlobalHistogramKMeans(object):
    """
    K-means tools for Global Histogram-Based Color Image Clustering
    with euclidean distance function.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.

    init : {'k-means++' or 'random'}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean tools in a
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

    distances_ : array
        Distances for each cluster.

    sum_distances_ : float
        Sum distances for all clusters.
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
        self.distances_ = np.zeros(self.__n_clusters, dtype=float)
        self.sum_distances_ = None

        self.__prev_distances = None

        self.__best_labels = None
        self.__best_cluster_centers = None
        self.__best_distances = None
        self.__best_sum_distances = None

    def __local_reset(self):
        """ Resets local state of the object. """
        self.labels_ = None
        self.cluster_centers_ = None
        self.distances_ = np.zeros(self.__n_clusters, dtype=float)
        self.sum_distances_ = None

        self.__prev_distances = None

    def __global_reset(self):
        """ Resets global state of the object. """
        self.random_state = np.random.RandomState(self.random_state_seed)
        self.__best_labels = None
        self.__best_cluster_centers = None
        self.__best_distances = None
        self.__best_sum_distances = None

    def __random_init(self, data):
        """ Initializes cluster centers randomly. """
        idx = self.random_state.permutation(data.shape[0])[:self.__n_clusters]
        self.cluster_centers_ = np.copy(data[idx])

    @staticmethod
    def __calculate_distances(data, cluster_centers):
        """ Calculates images' distances. """
        return np.sqrt(np.sum((data - cluster_centers) ** 2, axis=2))

    def __kmeans_pp_init(self, data):
        """ Initializes cluster centers using 'k-means++' algorithm. """
        self.cluster_centers_ = np.copy(data[np.random.choice(range(data.shape[0]), 1), :])
        ext_data = data[:, np.newaxis, :]
        while self.cluster_centers_.shape[0] < self.__n_clusters:
            distances = GlobalHistogramKMeans.__calculate_distances(ext_data, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
            min_distances = np.min(distances, axis=1)
            min_distances_sum = np.sum(min_distances)
            prob = min_distances / min_distances_sum
            self.cluster_centers_ = np.vstack([self.cluster_centers_, data[np.random.choice(
                range(data.shape[0]), 1, p=prob), :]])

    def __assign_labels(self, data):
        """ Assigns labels for data. """
        ext_data = data[:, np.newaxis, :]
        distances = GlobalHistogramKMeans.__calculate_distances(ext_data, self.cluster_centers_)
        self.labels_ = np.argmin(distances, axis=1)
        clusters_not_empty = np.isin(range(self.__n_clusters), self.labels_)
        while not np.all(clusters_not_empty):
            empty_clusters = np.logical_not(clusters_not_empty)
            self.cluster_centers_[empty_clusters] = np.copy(data[self.random_state.random_integers(
                0, len(data) - 1, np.sum(empty_clusters))])
            distances = GlobalHistogramKMeans.__calculate_distances(ext_data, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
            clusters_not_empty = np.isin(range(self.__n_clusters), self.labels_)

    def __update_centers(self, data):
        """ Computes new claster centers. """
        for i in range(self.__n_clusters):
            points = np.array([data[j] for j in range(len(data)) if self.labels_[j] == i])
            self.cluster_centers_[i] = np.mean(points, axis=0)

    def __update_distances(self, data):
        """ Calculates new distances. """
        for i in range(self.__n_clusters):
            points = np.array([data[j] for j in range(len(data)) if self.labels_[j] == i])
            ext_points = points[:, np.newaxis, :]
            cluster = self.cluster_centers_[i][np.newaxis, :]
            distances = GlobalHistogramKMeans.__calculate_distances(ext_points, cluster)
            self.distances_[i] = np.sum(distances)
        self.sum_distances_ = np.sum(self.distances_)

    def __is_progress(self):
        """ Checks whether to break the algorithm or continue. """
        if self.__prev_distances is None:
            return True
        prev_distances_sorted = np.sort(self.__prev_distances)
        distances_sorted = np.sort(self.distances_)
        return np.any(np.abs(prev_distances_sorted - distances_sorted) > self.__tol)

    def __update_best_solution(self):
        """ Updates the best solution if a new solution is better than previous ones. """
        if self.__best_sum_distances is None or self.__best_sum_distances > self.sum_distances_:
            self.__best_labels = np.copy(self.labels_)
            self.__best_cluster_centers = np.copy(self.cluster_centers_)
            self.__best_distances = np.copy(self.distances_)
            self.__best_sum_distances = self.sum_distances_

    def __load_best_solution(self):
        """ Loads the best solution calculated by the algorithm in 'n_init' runs. """
        self.labels_ = np.copy(self.__best_labels)
        self.cluster_centers_ = np.copy(self.__best_cluster_centers)
        self.distances_ = np.copy(self.__best_distances)
        self.sum_distances_ = self.__best_sum_distances

    def __predict_labels(self, data):
        """ Predicts final labels. """
        ext_data = data[:, np.newaxis, :]
        distances = GlobalHistogramKMeans.__calculate_distances(ext_data, self.cluster_centers_)
        self.labels_ = np.argmin(distances, axis=1)

    def fit_predict(self, data):
        """ Performs k-means algorithm for Local Histogram-Based Color Image Clustering task.
            Computes cluster centers and predicts cluster index for each sample. """
        self.__global_reset()
        for i in range(self.__n_init):
            #print "Init no: %s" % i
            self.__local_reset()
            if self.__init == 'k-means++':
                self.__kmeans_pp_init(data)
            elif self.__init == 'random':
                self.__random_init(data)
            else:
                raise ValueError('n_init: Inappropriate initialization method')
            for iteration in range(self.__max_iter):
                #print "Iteration no: %s" % iteration
                self.__assign_labels(data)
                self.__update_centers(data)
                self.__update_distances(data)
                if not self.__is_progress():
                    break
                self.__prev_distances = np.copy(self.distances_)
            self.__update_best_solution()
        self.__load_best_solution()
        self.__predict_labels(data)
        return self.labels_
