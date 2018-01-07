import numpy as np
import cv2
from itertools import chain
import os
import glob

import os
import fnmatch
from clustering.k_means import LocalHistogramKMeans

import shutil
import pathlib


def recursive_glob(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def get_image_paths(directory):
    image_paths = []
    for filename in glob.glob(directory + '*',):
        print filename
        if not os.path.isdir(filename):
            image_paths.append(filename)
    return image_paths


def read_images(image_paths):
    images = []
    for path in image_paths:
        if not os.path.isdir(path):
            rgb_img = cv2.imread(path, cv2.IMREAD_COLOR)
            #hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
            images.append(rgb_img)
    return images

#im = cv2.imread("/home/tomasz/Documents/Images/river/river13.tif")
#row, col= im.shape[:2]
#bottom= im[row-2:row, 0:col]
#mean= cv2.mean(bottom)[0]
directory = '/home/tomasz/Documents/Images/'

image_paths = recursive_glob(directory, '*')
print '\n'.join(image_paths)
thefile = open('test2.txt', 'w')
for item in image_paths:
  thefile.write("%s\n" % item)
thefile.close()

#image_paths = get_image_paths(directory)
images = read_images(image_paths)

print "Opened images"

N = 8
M = 8
S = 2

# Different applications use different scales for HSV. For example gimp uses H = 0-360, S = 0-100 and V = 0-100.
# But OpenCV uses H: 0 - 180, S: 0 - 255, V: 0 - 255

results = np.zeros((len(images), N * M, 3))

i_idx = 0
for img in images:
    print "Image nr: %s" % i_idx
    my_image=cv2.copyMakeBorder(img, top=S, bottom=S, left=S, right=S, borderType= cv2.BORDER_CONSTANT, value=[0, 0, 0])
    hsv = cv2.cvtColor(my_image, cv2.COLOR_BGR2HSV)
    #print hsv
    overlap_blocks = [hsv[i-S:i+32+S, j-S:j+32+S] for i in xrange(S, 256+S, 32) for j in xrange(S, 256+S, 32)]
    idx = 0
    for nimg in overlap_blocks:
        h, s, v = cv2.split(nimg)
        #print h
        hue = np.array(list(chain.from_iterable(h)))
        #print 'Hue before'
        #print hue
        #print 'Hue after'
        hue = np.round((hue / 180.0) * 255.0)
        #print hue
        #exit()
        sat = np.array(list(chain.from_iterable(s)))
        val = np.array(list(chain.from_iterable(v)))

        # ile ja mialem miec binsow???
        hue_hist, h_arg = np.histogram(hue, bins=256)
        sat_hist, s_arg = np.histogram(sat, bins=256)
        val_hist, v_arg = np.histogram(val, bins=256)
        h_m = np.argmax(hue_hist)
        s_m = np.argmax(sat_hist)
        v_m = np.argmax(val_hist)
        results[i_idx,idx,:] = np.array([h_arg[h_m], s_arg[s_m], v_arg[v_m]])
        idx = idx +1
    i_idx = i_idx + 1

del images

kmeans = LocalHistogramKMeans(20, 'k-means++', 10, 300, 1e-4)
kmeans(results)
print kmeans.cluster_centers_
print kmeans.labels_
print kmeans.sum_similarities_

res = zip(image_paths, kmeans.labels_)

thefile = open('test.txt', 'w')
for item in res:
  thefile.write("%s %s___\n" % item)
thefile.close()

# zapisywanie jeszcze nie dziala
result_directory = "results_local/"
if os.path.exists(result_directory):
    shutil.rmtree(result_directory)
for image_path, kmean in res:
    # Create directory for cluster, eg global/12 for cluster 12
    cluster_dir = result_directory + str(kmean)
    pathlib.Path(cluster_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, cluster_dir + '/' + os.path.basename(image_path))
