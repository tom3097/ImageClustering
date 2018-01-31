import time

import os

from Dataset import Dataset
from tools import GlobalHistogramKMeans
from tools import Pca
from Evaluator import Evaluator

root_path = '/home/tomasz/Documents/Images/'

# Creating dataset
dataset = Dataset()
dataset.create(root_path)

# Getting histogram from images.
histograms = dataset.get_histograms()

iterations = 300
dimensions_left = 400


def main(cluster_num):
    print "Basic global clustering"
    print "Iterations: " + str(iterations)
    print "Clusters: " + str(cluster_num)
    start = time.time()
    # Performing Global Histogram-Based Color Image Clustering
    globalHistogramKMeans = GlobalHistogramKMeans(cluster_num, 'k-means++', 10, 300, 1e-4, 0)
    predicted_labels = globalHistogramKMeans.fit_predict(histograms)
    end = time.time()
    print "Time: " + str(end - start)

    # Adequate to sklearn's inertia_
    print "Sum distances: %s" % globalHistogramKMeans.sum_distances_

    true_labels = dataset.get_true_labels()

    # Evaluation
    purity = Evaluator.calculate_purity(true_labels, predicted_labels)
    precision, recall = Evaluator.calculate_precision_recall(true_labels, predicted_labels)
    ri = Evaluator.calculate_ri(true_labels, predicted_labels)

    # Show results
    print "Purity: %s" % purity
    print "Precision: %s" % precision
    print "Recall: %s" % recall
    print "Random index value: %s" % ri

    dataset.write_results(os.path.join("results_global/", str(cluster_num), "no-pca"), predicted_labels)

    print "PCA global clustering"
    print "Dimensions left " + str(dimensions_left)
    print "Iterations: " + str(iterations)
    print "Clusters: " + str(cluster_num)


    start = time.time()
    # Changing dimensions from 768 to selected value.
    pca = Pca(dimensions_left)
    reduced_histograms = pca.fit_transform(histograms)

    # Clustering and predicting
    predicted_labels = globalHistogramKMeans.fit_predict(reduced_histograms)
    end = time.time()
    print "Time: " + str(end - start)

    # Adequate to sklearn's inertia_
    print "Sum distances: %s" % globalHistogramKMeans.sum_distances_

    # Evaluation
    purity = Evaluator.calculate_purity(true_labels, predicted_labels)
    precision, recall = Evaluator.calculate_precision_recall(true_labels, predicted_labels)
    ri = Evaluator.calculate_ri(true_labels, predicted_labels)

    # Show results
    print "Purity: %s" % purity
    print "Precision: %s" % precision
    print "Recall: %s" % recall
    print "Random index value: %s" % ri
    dataset.write_results(os.path.join("results_global/", str(cluster_num), "pca"), predicted_labels)


for i in xrange(15, 16):
    main(i)
    print ""
