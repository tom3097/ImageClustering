from Dataset import Dataset
from clustering.k_means import LocalHistogramKMeans
from Evaluator import Evaluator


root_path = '/home/tomasz/Documents/Images/'

dataset = Dataset()
dataset.create(root_path)

part_images = dataset.partition_images()

print part_images

localHistogramKMeans = LocalHistogramKMeans(21, 'k-means++', 10, 300, 1e-4)
localHistogramKMeans(part_images)
print(localHistogramKMeans.cluster_centers_)
print(localHistogramKMeans.labels_)
print(localHistogramKMeans.sum_similarities_)

true_labels = dataset.get_true_labels()

evaluator = Evaluator()
evaluator.calculate_purity(true_labels, localHistogramKMeans.labels_)
evaluator.calculate_rand_index(true_labels, localHistogramKMeans.labels_)


