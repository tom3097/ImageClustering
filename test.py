from clustering import GlobalHistogramKMeans
import numpy as np

rng = np.random.RandomState(2)

#data = np.zeros((50,8,3))
#for j in range(50):
#    row = np.zeros((8, 3))
#    for i in range(8):
#        r = [rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1)]
#        row[i] = r
#    data[j] = np.copy(row)

#kmeans = LocalHistogramKMeans(4, 'k-means++', 30, 100, 1e-4)
#kmeans(data)

#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#print(kmeans.sum_similarities_)

data = np.zeros((10, 4))
for j in range(10):
    row = np.zeros(4)
    for i in range(4):
        row[i] = rng.random_integers(0,255,1)
    data[j] = np.copy(row)

print data


ghk = GlobalHistogramKMeans(4, 'k-means++', 30, 100, 1e-4)