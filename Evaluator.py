"""
Allows evaluation of Global Histogram-Based Color Image Clustering and
Local Histogram-Based Color Image Clustering by calculating purity, precision,
recall and random index.

The idea comes from "Introduction to Information Retrieval":
https://nlp.stanford.edu/IR-book/pdf/16flat.pdf
"""

import scipy.special


class Evaluator(object):
    @staticmethod
    def calculate_purity(true_labels, predicted_labels):
        """ Calculates purity for clustering results. """
        label_no = max(predicted_labels) + 1
        total_count = 0

        for i in xrange(label_no):
            indexes = [idx for idx, x in enumerate(predicted_labels) if x == i]
            i_true_labels = true_labels[indexes].tolist()
            most_common = max(set(i_true_labels), key=i_true_labels.count)
            total_count += i_true_labels.count(most_common)

        purity = float(total_count) / len(predicted_labels)
        return purity

    @staticmethod
    def __calculate_tp_fp(true_labels, predicted_labels):
        """ Calculates 'True positive' and 'False positive' for clustering results. """
        label_no = max(predicted_labels) + 1
        tp_fp = 0
        tp = 0
        for i in xrange(label_no):
            indexes = [idx for idx, x in enumerate(predicted_labels) if x == i]

            indexes_len = len(indexes)
            if indexes_len < 2:
                continue
            tp_fp += scipy.special.binom(indexes_len, 2)

            i_true_labels = true_labels[indexes].tolist()
            set_i_true_labels = set(i_true_labels)
            for label in set_i_true_labels:
                true_labels_count = i_true_labels.count(label)
                if true_labels_count < 2:
                    continue
                tp += scipy.special.binom(true_labels_count, 2)

        fp = tp_fp - tp
        return [tp, fp]

    @staticmethod
    def __calculate_tp_fp_tn_fn(true_labels, predicted_labels):
        """ Calculates 'True positive', 'False positive', 'True negative' and
        'False negative' for clustering results. """
        label_no = max(predicted_labels) + 1

        tp_fp_tn_fn = (len(predicted_labels) * (len(predicted_labels) - 1)) / 2
        tp, fp = Evaluator.__calculate_tp_fp(true_labels, predicted_labels)
        tn_fn = tp_fp_tn_fn - tp - fp
        fn = 0

        for i in set(true_labels):
            item_count = {}
            for j in xrange(label_no):
                indexes = [idx for idx, x in enumerate(predicted_labels) if x == j]
                i_true_labels = true_labels[indexes].tolist()
                item_count[j] = i_true_labels.count(i)

            count_sum = sum(item_count.values())
            for j in xrange(label_no):
                count_sum -= item_count[j]
                fn += item_count[j] * count_sum

        tn = tn_fn - fn
        return [tp, fp, tn, fn]

    @staticmethod
    def calculate_precision_recall(true_labels, predicted_labels):
        """ Calculates precision and recall for clustering results. """
        tp, fp, tn, fn = Evaluator.__calculate_tp_fp_tn_fn(true_labels, predicted_labels)
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        return [precision, recall]

    @staticmethod
    def calculate_ri(true_labels, predicted_labels):
        """ Calculates random index for clustering results. """
        tp, fp, tn, fn = Evaluator.__calculate_tp_fp_tn_fn(true_labels, predicted_labels)
        return float(tp + tn) / (tp + fp + fn + tn)
