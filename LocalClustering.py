import time

import os

from Dataset import Dataset
from tools import LocalHistogramKMeans
from Evaluator import Evaluator

root_path = '/home/tomasz/Documents/STL-10/'

# Creating dataset
dataset = Dataset()
dataset.create(root_path)

# Getting partitioned images
part_images = dataset.partition_images()

iterations = 300


def main(cluster_num):
    print "Local clustering"
    print "Iterations: " + str(iterations)
    print "Clusters: " + str(cluster_num)
    start = time.time()
    # Performing Local Histogram-Based Color Image Clustering
    localHistogramKMeans = LocalHistogramKMeans(cluster_num, 'k-means++', 10, 300, 1e-4, 0)
    predicted_labels = localHistogramKMeans.fit_predict(part_images)
    end = time.time()
    print "Time: " + str(end - start)

    # Adequate to sklearn's inertia_
    print "Sum similarities: %s" % localHistogramKMeans.sum_similarities_

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

    dataset.write_results(os.path.join("results_local/", str(cluster_num)), predicted_labels)

for i in xrange(8, 13):
    main(i)
    print ""
