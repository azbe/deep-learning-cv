import glob
import os
import pdb
import tqdm

import keras
import numpy as np
from PIL import Image


def main(model_path, dataset_path):
    wrong = 0
    model = keras.models.load_model(model_path)    
    for filename in glob.glob(os.path.join(dataset_path, "*.png")):
        image = np.array(Image.open(filename))
        image = (image - 127.5) / 127.5
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = np.round(pred)
    


if __name__ == "__main__":
    main("data/model.hdf5", "data/validation/1")