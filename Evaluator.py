import scipy.special

class Evaluator(object):
    def __init__(self):
        pass

    def calculate_purity(self, true_labels, predicted_labels):
        print "Calculating purity..."
        label_no = max(predicted_labels) + 1

        total_count = 0

        for i in xrange(label_no):
            indexes = [idx for idx, x in enumerate(predicted_labels) if x == i]
            tt = true_labels[indexes].tolist()
            print tt
            print set(tt)
            max_count = max(set(tt), key=tt.count)
            print max_count
            c = tt.count(max_count)
            print c
            total_count += c

        purity = float(total_count) / len(predicted_labels)
        print "Purity %s" % purity

    def calculate_rand_index(self, true_labels, predicted_labels):
        # TP - oblicz ile kazde wystepuje
        print "Calculating random index..."
        label_no = max(predicted_labels) + 1

        x = len(predicted_labels)
        # total negatives plus total positives must be equal total_total
        total_total = x * (x-1) / 2

        TP = 0

        TP_and_FP = 0

        for i in xrange(label_no):
            indexes = [idx for idx, x in enumerate(predicted_labels) if x == i]
            tt = true_labels[indexes].tolist()

            c = len(indexes)
            if c < 2:
                continue
            TP_and_FP += scipy.special.binom(c, 2)

            s_tt = set(tt)
            for e in s_tt:
                c = tt.count(e)
                if c < 2:
                    continue
                TP += scipy.special.binom(c, 2)

        FP = TP_and_FP - TP

        total_negatives = total_total - TP_and_FP

        FN = 0

        for i in set(true_labels):
            d = {}
            for j in xrange(label_no):
                indexes = [idx for idx, x in enumerate(predicted_labels) if x == j]
                tt = true_labels[indexes].tolist()
                d[j] = tt.count(i)
            print d.values()
            val_sum = sum(d.values())
            print val_sum

            tmp_sum = 0

            for j in xrange(label_no):
                val_sum = val_sum - d[j]
                tmp_sum = tmp_sum + (d[j] * val_sum)

            FN += tmp_sum

        TN = total_negatives - FN


        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        RI = float(TP + TN) / (TP + FP + FN + TN)



        #FN - here comes the dragon

        print "TP = %s" % TP
        print "FP = %s" % FP
        print "TN = %s" % TN
        print "FN = %s" % FN
        print "Precision = %s" % precision
        print "Recall = %s" % recall
        print "RI = %s" % RI