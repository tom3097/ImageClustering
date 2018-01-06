"""Implementation of k-means algorithm
"""


# https://github.com/thilinamb/k-means-parallel/blob/master/src/KMeansPP.py kmeans++

import numpy as np
import math


#def KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None):
 #   pass

class KMeans2(object):
    def __init__(self, n_clusters, n_iter, n_init):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_init = n_init
        self.rng = np.random.RandomState(2)
        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(n_clusters, dtype=float)
        self.sum_similarities_ = None

        self.prev_labels = None
        self.prev_centers = None
        self.prev_similarity = None
        self.prev_sum_similarity = -1.0


    def __select_initial_centers(self, data):
        i = self.rng.permutation(data.shape[0])[:self.n_clusters]
        centers = data[i]
        return centers

    def calculate_hue_similarity(self, h_q, h_t):
        heuristic_k = 2
        #print '------'
        d = h_q - h_t
        #print d
        #print '---------'
        #k = np.linalg.norm(d, axis=1)
        # punkty w przestrzeni jednowymiarowej, wystarczy odjac i modulo
        k = np.absolute(d)
        #print k
        h_1 = k * 2.0 * math.pi / 256.0
        #print '-------'
        #print h_1
        #print '------------'
        h_2 = np.cos(h_1)
        #print h_2
        #print '-----'
        h_3 = np.power(h_2, 2)
        #print h_3
        #print '------'
        h_4 = (1 - h_3) / 2.0
        #print h_4
        return h_4

    def calculate_saturation_similarity(self, s_q, s_t):
        #print  s_q
        #print '----'
        #print s_t
        #print '----'
        d = s_q - s_t
        #print d
        #print '----'
        k = np.absolute(d)
        f = k / 256.0
        return f

    def calculate_value_similarity(self, v_q, v_t):
        d = v_q - v_t
        #print d
        #print '----'
        k = np.absolute(d)
        f = k / 256.0
        return f

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


    def cal_distances(self, data, centroids):
        #s = data - centroids
        #print s
        #print '------------'
        #print data.shape
        hue = data[range(data.shape[0]), :, 0]
        sat = data[range(data.shape[0]), :, 1]
        val = data[range(data.shape[0]), :, 2]
        #print hue
        #print centroids
        h_centroids = centroids[range(centroids.shape[0]), 0]
        s_centroids = centroids[range(centroids.shape[0]), 1]
        v_centroids = centroids[range(centroids.shape[0]), 2]
        #print h_centroids

        hh = self.calculate_hue_similarity(hue, h_centroids)
        ss = self.calculate_saturation_similarity(sat, s_centroids)
        vv = self.calculate_value_similarity(val, v_centroids)
        #print '--------'
        a = 2.5
        b = 0.5
        c = 0.0
        licznik = 1.0
        mianowniek = 1.0 + a * hh + b * ss + c * vv
        #print 'Mianownek'
        wynik = licznik / mianowniek
        return wynik


        #print h_centroids
        #hue_similarity = self.calculate_hue_similarity(hue, h_centroids)
        #print hue_similarity




    def k_means_pp_init(self, data):
        # https://github.com/thilinamb/k-means-parallel/blob/master/src/KMeansPP.py
        # wybieram pierwszy centroid czysto losowo
        self.cluster_centers_ = data[np.random.choice(range(data.shape[0]), 1), :]
        #self.cluster_centers_ = centroids
        ext_data = data[:, np.newaxis, :]
        #print '-----'
        #print self.cluster_centers_
        #print '------'
        #print ext_data
        #print '------'

        #centroids = centroids[:, np.newaxis, :]


        while self.cluster_centers_.shape[0] < self.n_clusters:
            #print 'DUPA'
            custom_distances = self.cal_distances(ext_data, self.cluster_centers_)
            #print 'Custom'
            #print custom_distances
            self.labels_ = np.argmax(custom_distances, axis=1)
            #print self.labels_
            biggest_similarities = np.max(custom_distances, axis=1)
            #print biggest_similarities

            # S przyjmuje wartosci od 0 do 1, wiec
            non_similarities = 1.0 - biggest_similarities
            #print non_similarities

            non_similarities_sum = np.sum(non_similarities)
            #print non_similarities_sum
            chose_prob = non_similarities / non_similarities_sum
            # prawdopodobienstwo tym wieksze, im wiekszy dystans ->prawdopodobienstwo tym wieksze, im mniejsze similarity
            #print chose_prob

            self.cluster_centers_ = np.vstack([self.cluster_centers_, data[np.random.choice(range(data.shape[0]), 1, p=chose_prob), :]])
            #self.cluster_centers_ = centroids
            #print self.cluster_centers_
            # 1. wyliczamy dystanse dla kazdego
            #.2. znajdujemy centroid dla kazdego
            # 3. sumujemy odleglosci od wszystkich punktow do ich centroidow
    # min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1 tam gdzie najmniejsze wartosci wstawiamy 1, dla kqzdego punktu
    # j_val = sumujemy te najmniejsze dystanse - pomocnicza wartosc





    def __assign_labels(self, data):
        ext_data = data[:, np.newaxis, :]
        custom_distances = self.cal_distances(ext_data, self.cluster_centers_)
        self.labels_ = np.argmax(custom_distances, axis=1)
        #print custom_distances
        #print self.labels_



    def __update_centers(self, data):
        for i in xrange(self.n_clusters):
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            #print 'points'
            #print points
            if points.size > 0:
                self.cluster_centers_[i] = np.mean(points, axis=0)
                #print self.cluster_centers_[i]
            else:
                self.cluster_centers_[i] = data[self.rng.random_integers(0, len(data)-1, 1)]


    def __update_similarities(self, data):
        for i in xrange(self.n_clusters):
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            mm = None
            #print '-----------777----'
            #print points
            if points.size == 0:
                mm = 0.0
                # tu chyba jakis nan by sie przydal
            else:
                points_ext = points[:, np.newaxis, :]
                #print self.cluster_centers_
                bb =  self.cluster_centers_[i][np.newaxis, :]
                ##print bb
                custom_distances = self.cal_distances(points_ext, bb)
                #print custom_distances
                #print '----------777-----'
                mm = np.mean(custom_distances)
                #print mm
                #print i
            #print mm
            self.similarities_[i] = mm
        self.sum_similarities_ = np.mean(self.similarities_)
        print self.sum_similarities_




    def __compare_centroids(self):
        """ jesli centroidy sie nie zmieniy od ostatniej rundy, to stabilized = True"""
        pass


    def __call__(self, data):

        self.prev_sum_similarity = -1.0
        self.prev_similarity = None
        self.prev_centers = None
        self.prev_labels = None

        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(self.n_clusters, dtype=float)
        self.sum_similarities_ = None

        for i in xrange(self.n_init):

            self.k_means_pp_init(data)
            #centers = self.__select_initial_centers(data)
            #exit()

            #print self.cluster_centers_


            for iterr in xrange(self.n_iter):
                self.__assign_labels(data)
                self.__update_centers(data)
                self.__update_similarities(data)
                #if iterr == 300:
                #   exit()
            if self.sum_similarities_ > self.prev_sum_similarity:
                self.prev_labels = self.labels_
                self.prev_centers = self.cluster_centers_
                self.prev_similarity = self.similarities_
                self.prev_sum_similarity = self.sum_similarities_


        self.labels_ = self.prev_labels
        self.cluster_centers_ = self.prev_centers
        self.similarities_ = self.prev_similarity
        self.sum_similarities_ = self.prev_sum_similarity

        return (self.labels_, self.cluster_centers_)













