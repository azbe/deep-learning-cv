import keras
import numpy as np
import sklearn.metrics


class F1ScoreCallback(keras.callbacks.Callback):
    def __init__(self, val_gen, steps, log_path):
        self.val_gen = val_gen
        self.steps = steps
        self.log_path = log_path
    
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for _ in range(self.steps):
            images, labels = next(self.val_gen)
            preds = self.model.predict(images)
            preds = np.squeeze(preds)
            preds = np.round(preds)
            y_true.extend(labels)
            y_pred.extend(preds)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        with open(self.log_path, "a") as f:
            f.write("%.4f\n" % f1)
        print("val_f1_score: %.4f" % f1)
