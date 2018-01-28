from Dataset import Dataset
from tools import LocalHistogramKMeans
from Evaluator import Evaluator

root_path = '/home/tomasz/Documents/Images/'

# Creating dataset
dataset = Dataset()
dataset.create(root_path)

# Getting partitioned images
part_images = dataset.partition_images()

# Performing Local Histogram-Based Color Image Clustering
localHistogramKMeans = LocalHistogramKMeans(21, 'k-means++', 10, 300, 1e-4, 0)
predicted_labels = localHistogramKMeans.fit_predict(part_images)

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
