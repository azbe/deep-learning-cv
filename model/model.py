import numpy as np
import keras
import tensorflow
import sys

from model.models import Models
from model.f1_score_callback import F1ScoreCallback
from util.logger import Logger

class Model:
    def __init__(self, model, dropout=None, bias_init=None, class_weights=None, optimizer=keras.optimizers.Adam, optimizer_args=None, metrics=None):
        self.model = model
        self.dropout = dropout
        self.bias_init = bias_init
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args if optimizer_args else {}
        self.class_weights = class_weights
        self.metrics = metrics if metrics else []
        self._logger = Logger(__name__)
    
    def build(self):
        self._logger.logger.info("Building model \"{}\"".format(str(self.model)))
        model = self.model.get_model()(include_top=False, weights=None)
        inputs, net = model.input, model.output
        
        if self.dropout:
            self._logger.logger.info("Adding dropout with keep_prob={}".format(self.dropout))
            net = keras.layers.Dropout(self.dropout)(net)

        net = keras.layers.GlobalAveragePooling2D()(net)

        self._logger.logger.info("Using bias initializer={} for last layer".format(self.bias_init if self.bias_init else 0.0))
        bias_initializer = keras.initializers.Constant([self.bias_init]) if self.bias_init else "zeros"
        net = keras.layers.Dense(units=1, 
                                 activation=keras.activations.sigmoid, 
                                 bias_initializer=bias_initializer)(net)
        
        self._logger.logger.info("Compiling model with optimizer {} with params: {}".format(
            str(self.optimizer),
            ", ".join(["{}={}".format(key, val) for key, val in self.optimizer_args.items()])))
        self._logger.logger.info("Adding metrics: {}".format(", ".join(map(str, self.metrics))))
        self.model = keras.models.Model(inputs=inputs, outputs=net)
        self.model.compile(
            optimizer=keras.optimizers.Adam(**self.optimizer_args),
            loss=keras.losses.binary_crossentropy,
            metrics=[metric.get_metric()(name=str(metric)) for metric in self.metrics])
    
    def train(self, train_gen, val_gen, epochs, save_path):
        try:
            class_weights = self.class_weights
            if self.class_weights:
                class_weights = {idx: class_weight for idx, class_weight in enumerate(self.class_weights)}
            self.model.fit_generator(
                train_gen, 
                steps_per_epoch=1,
                epochs=epochs,
                callbacks=[
                    F1ScoreCallback(val_gen, steps=125),
                    keras.callbacks.ModelCheckpoint(save_path, save_best_only=False),
                ],
                validation_data=val_gen,
                validation_steps=125,
                class_weight=class_weights,
                verbose=2)
        except KeyboardInterrupt:
            pass
