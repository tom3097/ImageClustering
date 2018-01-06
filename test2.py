from clustering import KMeans2
import numpy as np

rng = np.random.RandomState(2)
data = np.zeros((20, 3))
#print data


for i in xrange(20):
    d = [rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1)]
    data[i] = d

#print data

kmeans2 = KMeans2(4, 30, 100)
clusters, centers = kmeans2(data)
print clusters, centers

print '------------'
points = np.array([data[j] for j in xrange(len(data)) if clusters[j] == 0])
print points


points = np.array([data[j] for j in xrange(len(data)) if clusters[j] == 1])
print points

points = np.array([data[j] for j in xrange(len(data)) if clusters[j] == 2])
print points

points = np.array([data[j] for j in xrange(len(data)) if clusters[j] == 3])
print points

print kmeans2.sum_similarities_