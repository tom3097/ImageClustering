import cv2
import numpy as np

from itertools import chain


class Image(object):
    def __init__(self, path, true_label):
        self.path = path
        self.true_label = true_label

    def __open_with_cv2(self, hsv=True):
        raw_img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if hsv:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
        return raw_img

    def partition(self, m=8, n=8, s=2):
        print "Partitioning image: %s" % self.path
        raw_img = self.__open_with_cv2()

        new_shape0 = raw_img.shape[0] - raw_img.shape[0] % m
        new_shape1 = raw_img.shape[1] - raw_img.shape[1] % n

        if new_shape0 != raw_img.shape[0] or new_shape1 != raw_img.shape[1]:
            # shape[0] is height, shape[1] is width
            raw_img = cv2.resize(raw_img, (new_shape1, new_shape0))

        img_padding = cv2.copyMakeBorder(raw_img, top=s, bottom=s, left=s, right=s,
                                         borderType=cv2.BORDER_REPLICATE)

        shape0_offset = raw_img.shape[0] / m
        shape1_offset = raw_img.shape[1] / n
        overlapping_blocks = [img_padding[i - s:i + shape0_offset + s, j - s:j + shape1_offset + s]
                              for i in range(s, raw_img.shape[0] + s, shape0_offset)
                              for j in range(s, raw_img.shape[1] + s, shape1_offset)]

        part_img = np.zeros((n * m, 3))
        index = 0

        for single_block in overlapping_blocks:
            h, s, v = cv2.split(single_block)
            # hue range is [0,179]
            hue = np.array(list(chain.from_iterable(h)))
            hue = np.round((hue / 179.0) * 255.0)
            # saturation range is [0,255]
            sat = np.array(list(chain.from_iterable(s)))
            # value range is [0,255]
            val = np.array(list(chain.from_iterable(v)))

            hue_hist, hue_arg = np.histogram(hue, bins=256)
            sat_hist, sat_arg = np.histogram(sat, bins=256)
            val_hist, val_arg = np.histogram(val, bins=256)

            hue_max = np.argmax(hue_hist)
            sat_max = np.argmax(sat_hist)
            val_max = np.argmax(val_hist)

            part_img[index, :] = np.array([hue_arg[hue_max], sat_arg[sat_max], val_arg[val_max]])
            index += 1

        return part_img
