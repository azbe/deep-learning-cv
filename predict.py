import glob
import os
import pdb
import tqdm

import keras
import numpy as np
from PIL import Image


def main(model_path, dataset_path):
    model = keras.applications.InceptionV3(include_top=False, weights=None)
    inputs, net = model.input, model.output
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.GlobalAveragePooling2D()(net)
    net = keras.layers.Dense(1, activation=keras.activations.sigmoid, bias_initializer=keras.initializers.Constant([-1.748545323]))(net)
    model = keras.models.Model(inputs=inputs, outputs=net)
    model.load_weights(model_path)
    with open("preds.txt", "w") as f:
        f.write("id,class\n")
        for filename in tqdm.tqdm(glob.glob(os.path.join(dataset_path, "*.png"))):
            image = np.array(Image.open(filename))
            image = image * (1./255.)
            image = np.expand_dims(image, axis=0)
            pred = model.predict(image)
            pred = np.round(pred)
            f.write("{},{}\n".format(os.path.basename(filename.split(".")[0]), int(pred)))


if __name__ == "__main__":
    main("models/model-102.hdf5", "data/test")