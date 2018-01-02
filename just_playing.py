import cv2
import matplotlib.pyplot as plt
from itertools import chain
import peakutils
import numpy as np
from peakutils.plot import plot as pplot

from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print kmeans.labels_
print kmeans.predict([[0, 0], [4, 4]])
print kmeans.cluster_centers_

path = "/Users/tomaszbochenski/Desktop/Images/river/river13.tif"

img = cv2.imread(path, cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print hsv

h, s, v = cv2.split(hsv)

hue = list(chain.from_iterable(h))
sat = list(chain.from_iterable(s))
val = list(chain.from_iterable(v))

plt.interactive(False)

plt.figure()
x, y, z = plt.hist(val)
plt.title('Histogram H')
plt.xlabel('Wartosc H')
plt.ylabel('Czestosc')
plt.show()
#print y
#
indexes = peakutils.indexes(y)
#print(indexes)
#print(x[indexes], y[indexes])

plt.figure(figsize=(10,8))
plt.subplot(311)                             #plot in the first cell
plt.subplots_adjust(hspace=.5)
plt.title("Hue")
hy, hx, _ = plt.hist(np.ndarray.flatten(h), bins=180)
plt.subplot(312)                             #plot in the second cell
plt.title("Saturation")
sy, sx, _ = plt.hist(np.ndarray.flatten(s), bins=128)
plt.subplot(313)                             #plot in the third cell
plt.title("Luminosity Value")
vy, vx, _ = plt.hist(np.ndarray.flatten(v), bins=128)
#plt.show()

max_h = (hy[np.argmax(hy)], hx[np.argmax(hy)])
max_s = (sy[np.argmax(sy)], sx[np.argmax(sy)])
max_v = (vy[np.argmax(vy)], vx[np.argmax(vy)])

#print("Max h x=%s y=%s" % max_h)
#print("Max s x=%s y=%s" % max_s)
#print("Max v x=%s y=%s" % max_v)

cv2.waitKey(0)

print hsv[3,7]

print hsv.dtype

# non-overlapping
nonoverlap = [hsv[i:i+32, j:j+32] for i in xrange(0, 256, 32) for j in xrange(0, 256, 32)]

res = np.zeros((8,8,3), dtype=np.uint8)
# wektor cech - zapisujemy jako wektor
ress = np.zeros((2, 192), dtype=np.uint8)

print res

i = 0
j = 0
k = 0

#print res[1,1]
#res[2,3]['col1'] = 5
#print res[2,3]['col1']

for nimg in nonoverlap:
    #bgr = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    #cv2.imshow('image', bgr)
    #cv2.waitKey(0)
    h, s, v = cv2.split(nimg)
    #plt.figure(figsize=(10, 8))
    plt.subplot(311)  # plot in the first cell
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue")
    hy, hx, _ = plt.hist(np.ndarray.flatten(h), bins=180)
    plt.subplot(312)  # plot in the second cell
    plt.title("Saturation")
    sy, sx, _ = plt.hist(np.ndarray.flatten(s), bins=128)
    plt.subplot(313)  # plot in the third cell
    plt.title("Luminosity Value")
    vy, vx, _ = plt.hist(np.ndarray.flatten(v), bins=128)
    #plt.show()

    max_h = (hy[np.argmax(hy)], hx[np.argmax(hy)])
    max_s = (sy[np.argmax(sy)], sx[np.argmax(sy)])
    max_v = (vy[np.argmax(vy)], vx[np.argmax(vy)])
    print("Max h x=%s y=%s" % max_h)
    print("Max s x=%s y=%s" % max_s)
    print("Max v x=%s y=%s" % max_v)
    res[i,j][0] = np.int(max_h[1])
    res[i,j][1] = np.int(max_s[1])
    res[i,j][2] = np.int(max_v[1])
    j += 1
    if j == 8:
        j = 0
        i += 1
    ress[0,k] = np.int(max_h[1])
    k += 1
    ress[0,k] = np.int(max_s[1])
    k += 1
    ress[0,k] = np.int(max_v[1])
    k += 1

print hsv
print res
#bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
#cv2.imshow('image', bgr)
#cv2.waitKey(0)


print "---"
print ress

print "DUPA"
kmeans = KMeans(n_clusters=2, random_state=0).fit(ress)
print kmeans.labels_
print kmeans.cluster_centers_