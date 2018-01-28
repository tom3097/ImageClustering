import numpy as np

class KMeans2(object):
    def __init__(self, n_clusters, n_iter, n_init, tol):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_init = n_init
        self.rng = np.random.RandomState(2)
        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(n_clusters, dtype=float)
        self.sum_similarities_ = None

        self.prev_sum = None
        self.prev_similarities = None

        self.tol = tol

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
        h_1 = k * 2.0 * np.pi / 256.0
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
        #print '---'
        #print v_q
        #print v_t
        #print d
        #print '---'
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
        #print data[range(data.shape[0]), :, :, 0]
        hue = data[range(data.shape[0]), :, :, 0]
        sat = data[range(data.shape[0]), :, :, 1]
        val = data[range(data.shape[0]), :, :, 2]
        #print hue
        #print centroids
        #print 'HCEN'
        #print centroids
        h_centroids = centroids[range(centroids.shape[0]), :, 0]
        s_centroids = centroids[range(centroids.shape[0]), :, 1]
        v_centroids = centroids[range(centroids.shape[0]), :, 2]
        #print 'HCEN'
        #print h_centroids

        hh = self.calculate_hue_similarity(hue, h_centroids)
        #print hh
        ss = self.calculate_saturation_similarity(sat, s_centroids)
        vv = self.calculate_value_similarity(val, v_centroids)
        #print vv
        #print '--------'
        a = 2.5
        b = 0.5
        c = 0.0
        licznik = 1.0
        mianowniek = 1.0 + a * hh + b * ss + c * vv
        #print 'Mianownek'
        wynik = licznik / mianowniek
        #print wynik
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

        #print self.cluster_centers_.shape[0]
        while self.cluster_centers_.shape[0] < self.n_clusters:
            #print 'DUPA'
            custom_distances = self.cal_distances(ext_data, self.cluster_centers_)
            #print custom_distances
            # tu suma ogolna

            a = np.mean(custom_distances, axis=2)
            #print a
            custom_distances = a

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


    def handle_empty_labels2(self, data):
        correct = False
        while correct is False:
            correct = True
            self.__assign_labels(data)
            for i in xrange(self.n_clusters):
                if i not in self.labels_:
                    self.cluster_centers_[i] = data[self.rng.random_integers(0, len(data) - 1, 1)]
                    correct = False
                    print 'Handle empty %s' % i
                    break


    def __assign_labels(self, data):
        ext_data = data[:, np.newaxis, :]
        custom_distances = self.cal_distances(ext_data, self.cluster_centers_)
        #print custom_distances
        custom_distances = np.mean(custom_distances, axis=2)
        self.labels_ = np.argmax(custom_distances, axis=1)
        res = np.isin(range(self.n_clusters), self.labels_)
        while not np.all(res):
            print 'Handling empty'
            self.cluster_centers_[res] = data[self.rng.random_integers(0, len(data) - 1, np.sum(res))]
            custom_distances = self.cal_distances(ext_data, self.cluster_centers_)
            custom_distances = np.mean(custom_distances, axis=2)
            self.labels_ = np.argmax(custom_distances, axis=1)
            res = np.isin(range(self.n_clusters), self.labels_)

        #print custom_distances
        #print self.labels_

    def handle_empty_cluster(self, index, data):
        self.cluster_centers_[index] = data[self.rng.random_integers(0, len(data) - 1, 1)]
        # sprawdz czy nadal jest jakis pusty
        last_empty = index
        empty_clusters = True
        while empty_clusters:
            #print 'Handling empt2222y %s' % last_empty
            empty_clusters = False
            # przyporzadkuj od nowa labelki
            self.__assign_labels(data)
            for i in xrange(self.n_clusters):
                points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
                if points.size == 0:
                    self.cluster_centers_[i] = data[self.rng.random_integers(0, len(data) - 1, 1)]
                    empty_clusters = True
                    last_empty = i
                    break



    def __update_centers(self, data):
        for i in xrange(self.n_clusters):
            #print self.labels_
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            #print 'points'
            #print points
            if points.size > 0:
                #print points
                #print '---'
                #print self.cluster_centers_[i]
                #print '----AAA'
                self.cluster_centers_[i] = np.mean(points, axis=0)
                #print self.cluster_centers_[i]
                #print '---'
                #print self.cluster_centers_[i]
            else:
                #print 'WTF'
                exit()
                #self.handle_empty_cluster(i, data)
                #self.cluster_centers_[i] = data[self.rng.random_integers(0, len(data)-1, 1)]
                # pusty klaster -> dodanie nowego nie pogorszy jakosci
                # po dodaniu od nowa trzeba przyporzadkowac labelki -> robimy to do moementu,
                # az po przyporzadkowaniu wszystko jest ok
                #self.handle_empty_cluster()


    def __update_similarities(self, data):
        for i in xrange(self.n_clusters):
            points = np.array([data[j] for j in xrange(len(data)) if self.labels_[j] == i])
            mm = None
            #print '-----------777----'
            #print points
            points_ext = points[:, np.newaxis, :]
            #print self.cluster_centers_
            bb =  self.cluster_centers_[i][np.newaxis, :]
            ##print bb
            custom_distances = self.cal_distances(points_ext, bb)
            custom_distances = np.mean(custom_distances, axis=2)
            #print '---'
            #print points
            #print custom_distances
            #print custom_distances
            #print '----------777-----'
            mm = np.mean(custom_distances)
            #print mm
            #print i
            #print mm
            self.similarities_[i] = mm
        self.sum_similarities_ = np.mean(self.similarities_)
        #print self.sum_similarities_




    def __compare_centroids(self):
        """ jesli centroidy sie nie zmieniy od ostatniej rundy, to stabilized = True"""
        pass

    def check_if_break(self):
        if self.prev_similarities is None:
            return False
        #print self.prev_similarities
        #print self.similarities_

        prev_sort = np.sort(self.prev_similarities)
        curr_sort = np.sort(self.similarities_)
        #print prev_sort
        #print curr_sort

        diff_sim = np.abs(prev_sort - curr_sort)
        #print diff_sim
        a =  diff_sim <= self.tol
        #print a
        s = np.all(a)
        #print diff_sim
        #print s
        return s

    def __call__(self, data):

        self.prev_sum_similarity = -1.0
        self.prev_similarity = None
        self.prev_centers = None
        self.prev_labels = None

        self.prev_sum = None

        self.labels_ = None
        self.cluster_centers_ = None
        self.similarities_ = np.zeros(self.n_clusters, dtype=float)
        self.sum_similarities_ = None

        for i in xrange(self.n_init):

            self.k_means_pp_init(data)
            #print self.cluster_centers_
            #centers = self.__select_initial_centers(data)
            #exit()

            #print self.cluster_centers_

            for iterr in xrange(self.n_iter):
                self.__assign_labels(data)
                #self.handle_empty_labels2(data)
                self.__update_centers(data)
                self.__update_similarities(data)

                #if self.prev_sum is not None:
                #    diff = np.abs(self.prev_sum - self.sum_similarities_)
                #    if diff <= self.tol:
                #        print 'Tolerance'
                #        break
                #if self.check_progress():
                #    break
                if self.check_if_break():
                    print 'COCOCOCOCOCOCOCOOOCOCOCOC'
                    break

                self.prev_sum = self.sum_similarities_
                self.prev_similarities = np.copy(self.similarities_)

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










class GlobalHistogramKMeans(object):
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
        """Resets local state of the object."""
        self.labels_ = None
        self.cluster_centers_ = None
        self.distances_ = np.zeros(self.__n_clusters, dtype=float)
        self.sum_distances_ = None

        self.__prev_distances = None

    def __global_reset(self):
        """Resets global state of the object."""
        self.random_state = np.random.RandomState(self.random_state_seed)
        self.__best_labels = None
        self.__best_cluster_centers = None
        self.__best_distances = None
        self.__best_sum_distances = None

    @staticmethod
    def __calculate_similarities(d_q, d_t):
        tmp1 = d_q - d_t
        tmp2 = tmp1 ** 2
        tmp3 = np.sum(tmp2)
        res = np.sqrt(tmp3)
        return res

    def __random_init(self, data):
        """Initializes cluster centers randomly."""
        idx = self.random_state.permutation(data.shape[0])[:self.__n_clusters]
        self.cluster_centers_ = np.copy(data[idx])




    def kmeans_pp_init(self, data):
        """Initializes cluster centers using 'k-means++' algorithm."""
        self.cluster_centers_ = data[np.random.choice(list(range(data.shape[0])), 1)]


        ext_data = data[:, np.newaxis, :]
        while self.cluster_centers_.shape[0] < self.__n_clusters:


            distances = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)


            self.labels_ = np.argmin(distances, axis=1)
            min_distances = np.min(distances, axis=1)
            min_distances_sum = np.sum(min_distances)
            prob = min_distances / min_distances_sum


            non_similarities = 1.0 - max_similarities
            non_similarities_sum = np.sum(non_similarities)
            prob = non_similarities / non_similarities_sum
            self.cluster_centers_ = np.vstack([self.cluster_centers_, data[np.random.choice(
                list(range(data.shape[0])), 1, p=prob), :]])

    def __assign_labels(self, data):
        """Assigns labels for data."""
        ext_data = data[:, np.newaxis, :]
        similarities = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)
        self.labels_ = np.argmax(similarities, axis=1)
        clusters_not_empty = np.isin(list(range(self.__n_clusters)), self.labels_)
        while not np.all(clusters_not_empty):
            empty_clusters = np.logical_not(clusters_not_empty)
            self.cluster_centers_[empty_clusters] = data[self.random_state.random_integers(
                0, len(data) - 1, np.sum(empty_clusters))]
            similarities = LocalHistogramKMeans.__calculate_similarities(ext_data, self.cluster_centers_)
            self.labels_ = np.argmax(similarities, axis=1)
            clusters_not_empty = np.isin(list(range(self.__n_clusters)), self.labels_)



