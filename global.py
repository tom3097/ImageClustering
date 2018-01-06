import sys
import os
import cv2
import glob
from itertools import chain
import pickle
import shutil
import numpy as np
import pathlib

from sklearn.cluster import KMeans

counter = 1
def img_to_combined_histogram(img):
    global counter
    print("Converting image " + str(counter))
    counter = counter + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hue = list(chain.from_iterable(h))
    sat = list(chain.from_iterable(s))
    val = list(chain.from_iterable(v))

    hue_hist, _ = np.histogram(hue, bins=256)
    sat_hist, _ = np.histogram(sat, bins=256)
    val_hist, _ = np.histogram(val, bins=256)

    return np.array(hue_hist + sat_hist + val_hist)


def get_image_paths(directory):
    image_paths = []
    for filename in glob.iglob(directory + '**/**', recursive=True):
        if not os.path.isdir(filename):
            image_paths.append(filename)
    return image_paths


def read_images(image_paths):
    images = []
    for path in image_paths:
        if not os.path.isdir(path):
            images.append(cv2.imread(path, cv2.IMREAD_COLOR))
    return images


def main(argv):
    if len(argv) == 1:
        print("Provide directory with images.")
        return

    directory = sys.argv[1]
    image_paths = get_image_paths(directory)
    images = read_images(image_paths)

    # Read histograms from cache if it's available
    if os.path.exists("cache"):
        with open("cache", "rb") as fp:
            histograms = pickle.load(fp)
    else:
        histograms = [img_to_combined_histogram(img) for img in images]
        # Save histograms to not recalculate them next time
        with open("cache", "wb") as fp:
            pickle.dump(histograms, fp)

    if len(argv) == 3:
        n_clusters = int(sys.argv[2])
    else:
        n_clusters = 21

    kmeans = KMeans(n_clusters=n_clusters).fit_predict(histograms)

    result_directory = "results_global/"
    if os.path.exists(result_directory):
        shutil.rmtree(result_directory)
    for kmean, image_path in zip(kmeans, image_paths):
        # Create directory for cluster, eg global/12 for cluster 12
        cluster_dir = result_directory + str(kmean)
        pathlib.Path(cluster_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, cluster_dir + '/' + os.path.basename(image_path))


if __name__ == "__main__":
    main(sys.argv)
