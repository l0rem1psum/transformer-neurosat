import numpy as np


class ConfusionMatrix(object):
    def __init__(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def add(self, other):
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        self.tp += other.tp

    def update_one(self, actual, predicted):
        if (not actual) and (not predicted):
            self.tn += 1
        elif (not actual) and predicted:
            self.fp += 1
        elif actual and (not predicted):
            self.fn += 1
        else:
            assert (actual and predicted)
            self.tp += 1

    def update(self, actuals, predicteds):
        for i in range(len(actuals)):
            self.update_one(actuals[i], predicteds[i])

    def pretty_print(self):
        formatted_table = '''       
+--------------------+-----------------+-------------------+
|                    | Actually is_sat | Actually is_unsat |
+--------------------+-----------------+-------------------+
| Predicted is_sat   |{:17d}|{:19d}|
| Predicted is_unsat |{:17d}|{:19d}|
+--------------------+-----------------+-------------------+
| Sensitivity (TPR)  |{:37.6f}|
+--------------------+-----------------+-------------------+
| Specificity (TNR)  |{:37.6f}|
+--------------------+-----------------+-------------------+
| Precision (PPV)    |{:37.6f}|
+--------------------+-----------------+-------------------+
| F-1 Score          |{:37.6f}|
+--------------------+-----------------+-------------------+
| Overall Accuracy   |{:37.6f}|
+--------------------+-----------------+-------------------+
        '''.format(
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            self.tpr(),
            self.tnr(),
            self.ppv(),
            self.f1(),
            self.accuracy())
        print(formatted_table)

    def tpr(self):
        """
        TPR = TP/P = TP / (TP+FN)
        :return: sensitivity, recall, hit rate, or true positive rate (TPR)
        """
        numerator = self.tp
        denominator = self.tp + self.fn
        if denominator == 0:
            return -1
        return numerator/denominator

    def sensitivity(self):
        """
        TPR = TP/P = TP / (TP+FN)
        :return: sensitivity, recall, hit rate, or true positive rate (TPR)
        """
        return self.tpr()

    def tnr(self):
        """
        TNR = TN/N = TN / (TN+FP)
        :return: specificity, selectivity or true negative rate (TNR)
        """
        numerator = self.tn
        denominator = self.tn + self.fp
        if denominator == 0:
            return -1
        return numerator / denominator

    def specificity(self):
        """
        TNR = TN/N = TN / (TN+FP)
        :return: specificity, selectivity or true negative rate (TNR)
        """
        return self.tnr()

    def ppv(self):
        """
        PPV = TP / (TP + FP) = 1 - FDR
        :return: precision or positive predictive value (PPV)
        """
        numerator = self.tp
        denominator = self.tp + self.fp
        if denominator == 0:
            return -1
        return numerator / denominator

    def precision(self):
        """
        PPV = TP / (TP + FP) = 1 - FDR
        :return: precision or positive predictive value (PPV)
        """
        return self.ppv()

    def npv(self):
        """
        NPV = TN / (TN + FN) = 1 - FOR
        :return: negative predictive value (NPV)
        """
        numerator = self.tn
        denominator = self.tn + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

    def f1(self):
        """
        F_1 = 2 * (PPV * TPR) / (PPV + TPR) = 2*TP / (2*TP + FP + FN)
        :return: F1 Score
        """
        numerator = 2 * self.tp
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

    def accuracy(self):
        """
        ACC = (TP + TN) / (P + N) = (TP + TN) / (TP + TN + FP + FN)
        :return: Accuracy
        """
        numerator = self.tp + self.tn
        denominator = self.tp + self.tn + self.fp + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

    def get_percentages(self):
        total = self.tn + self.fp + self.fn + self.tp
        assert (total > 0)
        matrix = ConfusionMatrix()
        matrix.tn = float(self.tn) / total
        matrix.fp = float(self.fp) / total
        matrix.fn = float(self.fn) / total
        matrix.tp = float(self.tp) / total
        return matrix

    def __str__(self):
        return str((self.tn, self.fp, self.fn, self.tp))

    def __repr__(self):
        return self.__str__()
