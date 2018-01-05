"""Implementation of k-means algorithm
"""

import numpy as np
import math


#def KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None):
 #   pass

class KMeans2(object):
    def __init__(self, n_clusters, n_iter):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.rng = np.random.RandomState(2)

    def __select_initial_centers(self, data):
        i = self.rng.permutation(data.shape[0])[:self.n_clusters]
        centers = data[i]
        return centers

    def calculate_hue_similarity(self, h_q, h_t):
        heuristic_k = 2
        h_1 = np.linalg.norm(h_q - h_t) * 2.0 * math.pi / 256.0
        h_2 = math.cos(h_1)
        h_3 = math.pow(h_2, heuristic_k)
        h_4 = (1 - h_3) / 2.0
        return h_4

    def calculate_saturation_similarity(self, s_q, s_t):
        return np.linalg.norm(s_q - s_t) / 256.0

    def calculate_value_similarity(self, v_q, v_t):
        return np.linalg.norm(v_q - v_t)

    """ Zakladam ze block_q = (h_q, s_q, v_q), block_t = (h_t, s_t, v_t)
    """
    def calculate_block_similarity(self, block_q, block_t):
        a = 2.5
        b = 0.5
        c = 0.0
        dh = self.calculate_hue_similarity(block_q[0], block_t[0])
        ds = self.calculate_saturation_similarity(block_q[1], block_t[1])
        dv = self.calculate_value_similarity(block_t[2], block_q[2])

        licznik = 1.0
        mianowniek = 1.0 + a * dh + b * ds + c * dv

        s = licznik / mianowniek
        if np.isnan(a):
            print 'NAN'
        return s



    def __assign_labels(self, data, centers):

        clusters = np.zeros(len(data))
        #print centers


        for i in xrange(len(data)):
            dist = []
            #print centers
            for j in xrange(len(centers)):
                d = self.calculate_block_similarity(data[i], centers[j])
                #print data[i]
                dist.append(d)
            cluster = np.argmin(dist)
            #print dist
            #print cluster
            clusters[i] = cluster

            #if i == len(data) - 1:
            #    exit()

        return clusters

    def __update_centers(self, data, clusters):
        centers = np.zeros((self.n_clusters, 3), dtype=float)
        #print centers
        for i in xrange(self.n_clusters):
            #print i
            points = np.array([data[j] for j in xrange(len(data)) if clusters[j] == i])
            if points.size > 0:
                centers[i] = np.mean(points, axis=0)
            else:
                centers[i] = data[self.rng.random_integers(0, len(data)-1, 1)]
            #print centers[i]
        #print centers
        return centers


    def __call__(self, data):
        centers = self.__select_initial_centers(data)
        print centers

        clusters = None

        for iterr in xrange(self.n_iter):
            clusters = self.__assign_labels(data, centers)
            #print clusters
            #exit()
            centers = self.__update_centers(data, clusters)
            print centers
            #if iterr == 300:
             #   exit()

        return (clusters, centers)













