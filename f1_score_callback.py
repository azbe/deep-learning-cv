import keras
import numpy as np
import sklearn.metrics


class F1ScoreCallback(keras.callbacks.Callback):
    def __init__(self, val_gen, steps):
        self.val_gen = val_gen
        self.steps = steps
    
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for _ in range(self.steps):
            images, labels = next(self.val_gen)
            preds = self.model.predict(images)
            preds = np.squeeze(preds)
            preds = np.round(preds)
            y_true.extend(labels)
            y_pred.extend(preds)
        print("val_f1_score: %.4f" % sklearn.metrics.f1_score(y_true, y_pred))
