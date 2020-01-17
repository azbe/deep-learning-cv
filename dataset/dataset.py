import os

import numpy as np
import keras

from util.logger import Logger


class Dataset:
    def __init__(self, path):
        self._logger = Logger(__name__)
        self.path = path
        self.train_path = os.path.join(path, "train")
        self.val_path = os.path.join(path, "validation")

    def build(self):
        self._logger.logger.info("Loading train dataset from \"{}\"".format(self.train_path))
        self.train_gen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255.,
            horizontal_flip=True,
            vertical_flip=True)
        self.train_gen = self.train_gen.flow_from_directory(
            self.train_path, 
            batch_size=32,
            target_size=(224, 224),
            class_mode="binary",
            shuffle=True)

        self._logger.logger.info("Loading validation dataset from \"{}\"".format(self.val_path))
        self.val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        self.val_gen = self.val_gen.flow_from_directory(
            self.val_path, 
            batch_size=32,
            target_size=(224, 224),
            class_mode="binary",
            shuffle=True)

