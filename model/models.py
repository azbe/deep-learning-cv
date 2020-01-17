from enum import Enum

import keras


class Models(Enum):
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    INCEPTIONV3 = "inceptionv3"

    def __str__(self):
        return self.value
    
    def get_model(self):
        if self == Models.RESNET50:
            return keras.applications.ResNet50
        if self == Models.RESNET101:
            return keras.applications.ResNet101
        if self == Models.INCEPTIONV3:
            return keras.applications.InceptionV3