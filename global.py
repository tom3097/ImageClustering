import sys
import os
import cv2
import glob2
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

    return np.append(np.append(hue_hist, sat_hist), val_hist)


def get_image_paths(directory):
    paths = glob2.glob(directory + '**')
    return [path for path in paths if not os.path.isdir(path)]


def read_images(image_paths):
    images = []
    for path in image_paths:
        if not os.path.isdir(path):
            images.append(cv2.imread(path, cv2.IMREAD_COLOR))
    return images


def get_params(argv):
    directory = n_clusters = None
    if len(argv) > 1:
        directory = sys.argv[1]
    if len(argv) > 2:
        n_clusters = int(sys.argv[2])
    return directory, n_clusters


def get_histograms(images):
    # Read histograms from cache if it's available
    if os.path.exists("cache_normal"):
        with open("cache_normal", "rb") as fp:
            histograms = pickle.load(fp)
    else:
        histograms = np.array([img_to_combined_histogram(img) for img in images])
        # Save histograms to not recalculate them next time
        with open("cache_normal", "wb") as fp:
            pickle.dump(histograms, fp)
    return histograms


def get_reduced_histograms(normal_histograms, new_dimension_num):
    # Read reduced histograms from cache if it's available
    if os.path.exists("cache_pca"):
        with open("cache_pca", "rb") as fp:
            reduced_histograms = pickle.load(fp)
    else:
        original_dimension_num = normal_histograms.shape[1]
        normal_histograms_transformed = normal_histograms.T
        cov_mat = np.cov([normal_histograms_transformed[i, :] for i in range(0, original_dimension_num)])

        # Calculate eigenvalues and eigenvectors
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(original_dimension_num)]

        # Sort eig_pairs by eigenvalues
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        matrix_w = np.hstack((eig_pairs[i][1].reshape(original_dimension_num, 1) for i in range(new_dimension_num)))

        reduced_histograms = matrix_w.T.dot(normal_histograms_transformed).T
        assert reduced_histograms.shape == (normal_histograms.shape[0], new_dimension_num)

        # Save histograms to not recalculate them next time
        with open("cache_pca", "wb") as fp:
            pickle.dump(reduced_histograms, fp)

    return reduced_histograms


def basic_kmeans(histograms, image_paths, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit_predict(histograms)
    write_results('results_global/basic/', kmeans, image_paths)


def pca_kmeans(histograms, image_paths, n_clusters):
    histograms_pca = get_reduced_histograms(histograms, 600)
    kmeans = KMeans(n_clusters=n_clusters).fit_predict(histograms_pca)
    write_results('results_global/pca/', kmeans, image_paths)


def write_results(result_directory, kmeans, image_paths):
    if os.path.exists(result_directory):
        shutil.rmtree(result_directory)
    for kmean, image_path in zip(kmeans, image_paths):
        # Create directory for cluster, eg global/12 for cluster 12
        cluster_dir = result_directory + str(kmean)
        pathlib.Path(cluster_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, cluster_dir + '/' + os.path.basename(image_path))


def main(argv):
    directory, n_clusters = get_params(argv)
    if directory is None:
        print("Provide directory with images.")
        return

    image_paths = get_image_paths(directory)
    images = read_images(image_paths)
    histograms = get_histograms(images)

    if n_clusters is None:
        n_clusters = 21

    basic_kmeans(histograms, image_paths, n_clusters)
    pca_kmeans(histograms, image_paths, n_clusters)


if __name__ == "__main__":
    main(sys.argv)
