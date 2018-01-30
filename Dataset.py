"""
Groups images and allows to perform non-trivial operations on a collection of images.
Implements operations crucial to experiments and data pre-processing.
"""

import os
import glob
import random
import shutil

import numpy as np
import pickle

import pathlib

from Image import Image


class Dataset(object):
    """
    Represents a dataset of images.
    """
    def __init__(self):
        self.labels = None
        self.images = []

    def create(self, root_path):
        """ Scans the root_path and creates a labeled collection based
            on the directory names. """
        self.labels = [dI for dI in os.listdir(root_path) if
                       os.path.isdir(os.path.join(root_path, dI))]
        for label in self.labels:
            path_pattern = os.path.join(root_path, label, '*')
            img_paths = glob.glob(path_pattern)
            imgs = [Image(path, label) for path in img_paths]
            self.images.extend(imgs)

    def partition_images(self, m=8, n=8, s=2):
        """ Partitions all images into m*n blocks. """
        if os.path.exists('local_cache'):
            with open('local_cache', 'rb') as fp:
                part_images = pickle.load(fp)
            return part_images

        print "Partitioning: this may take a while..."
        part_images = np.zeros((len(self.images), m * n, 3))
        index = 0
        for img in self.images:
            part_images[index, :] = img.partition(m, n, s)
            index += 1

        with open('local_cache', 'wb') as fp:
            pickle.dump(part_images, fp)

        return part_images

    def get_histograms(self):
        """ Calculates hue, saturation and value histograms for each image
            in the database. """
        if os.path.exists('global_cache'):
            with open('global_cache', 'rb') as fp:
                histograms = pickle.load(fp)
            return histograms

        histograms = np.array([img.get_histogram() for img in self.images])
        with open('global_cache', 'wb') as fp:
            pickle.dump(histograms, fp)

        return histograms

    def get_true_labels(self):
        """ Returns true labels for images in database. """
        return np.array([l.true_label for l in self.images])

    def write_results(self, result_directory, predicted_labels):
        for calculated_label, image in zip(predicted_labels, self.images):
            # Create directory for cluster, eg global/12 for cluster 12
            cluster_dir = os.path.join(result_directory, str(calculated_label))
            if not os.path.exists(cluster_dir):
                pathlib.Path(cluster_dir).mkdir(parents=True)
            shutil.copyfile(image.path, os.path.join(cluster_dir, os.path.basename(image.path)))
