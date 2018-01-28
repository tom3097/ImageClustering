import os
import glob
import random
import numpy as np

from Image import Image


class Dataset(object):
    def __init__(self):
        self.labels = None
        self.images = []

    def create(self, root_path):
        self.labels = [dI for dI in os.listdir(root_path) if
                       os.path.isdir(os.path.join(root_path, dI))]
        for label in self.labels:
            path_pattern = os.path.join(root_path, label, '*')
            img_paths = glob.glob(path_pattern)
            imgs = [Image(path, label) for path in img_paths]
            self.images.extend(imgs)
        random.shuffle(self.images)

    def partition_images(self, m=8, n=8, s=2):
        print "Performing partitioning. This may take a while..."
        part_images = np.zeros((len(self.images), m * n, 3))
        index = 0
        for img in self.images:
            part_images[index, :] = img.partition(m, n, s)
            index += 1
            #if index == 500:
            #    break
        return part_images

    def get_true_labels(self):
        return np.array([l.true_label for l in self.images])
