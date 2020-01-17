from enum import Enum

import keras


class Metrics(Enum):
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    TRUE_NEGATIVES = "true_negatives"
    FALSE_NEGATIVES = "false_negatives"
    RECALL = "recall"
    PRECISION = "precision"
    ACCURACY = "accuracy"
    AUC = "auc"

    def __str__(self):
        return self.value
    
    def get_metric(self):
        if self == Metrics.TRUE_POSITIVES:
            return keras.metrics.TruePositives
        if self == Metrics.FALSE_POSITIVES:
            return keras.metrics.FalsePositives
        if self == Metrics.TRUE_NEGATIVES:
            return keras.metrics.TrueNegatives
        if self == Metrics.FALSE_NEGATIVES:
            return keras.metrics.FalseNegatives
        if self == Metrics.RECALL:
            return keras.metrics.Recall
        if self == Metrics.PRECISION:
            return keras.metrics.Precision
        if self == Metrics.ACCURACY:
            return keras.metrics.BinaryAccuracy
        if self == Metrics.AUC:
            return keras.metrics.AUC
