from clustering import KMeans2
import numpy as np

rng = np.random.RandomState(2)
#print data

s = np.zeros((50,8,3))
for j in range(50):
    data = np.zeros((8, 3))
    for i in xrange(8):
        d = [rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1), rng.random_integers(0, 255, 1)]
        data[i] = d
    s[j] = np.copy(data)

print s
#print data

kmeans2 = KMeans2(4, 30, 100, 1e-4)
clusters, centers = kmeans2(s)
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