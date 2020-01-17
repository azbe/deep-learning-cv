import argparse
import glob
import os
import pdb
import sys
import tqdm

import keras
import tensorflow

from model.models import Models
from model.metrics import Metrics
from model.model import Model
from util.datetime import timestamp
from util.datetime import timestamp

import keras
import numpy as np
from PIL import Image


def main(data_dir, model, dropout, bias_init, learning_rate, class_weights, metrics, weights_path, output_path):
    model = Model(model, dropout, bias_init, class_weights, learning_rate, metrics)
    model.build()
    model.load_weights(weights_path)
    with open(output_path, "w") as f:
        f.write("id,class\n")
        for filename in tqdm.tqdm(glob.glob(os.path.join(data_dir, "*.png"))[:10]):
            image = np.array(Image.open(filename))
            image = image * (1./255.)
            image = np.expand_dims(image, axis=0)
            pred = model.predict(image)
            pred = np.round(pred)
            f.write("{},{}\n".format(os.path.basename(filename).split(".")[0], int(pred)))
    print("Output saved to \"{}\"".format(output_path))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="./data/test", help="Path to directory containing test images.")
    parser.add_argument("--model", type=Models, choices=list(Models), default=Models.RESNET50, help="Which base model to use")
    parser.add_argument("--dropout", type=float, help="Dropout rate (leave blank for no dropout)")
    parser.add_argument("--bias_init", type=float, help="Bias initializer to use for the last layer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate to use")
    parser.add_argument("--class_weights", type=float, nargs=2, help="Class weights to use for training")
    parser.add_argument("--metrics", type=Metrics, choices=list(Metrics), nargs="*", help="Which metrics to use for evaluation (F1 is on by default)")
    parser.add_argument("--weights_path", type=str, help="Path to load model weights from")
    parser.add_argument("--output_path", type=str, default="./output/test-{}.txt".format(timestamp()), help="Path to save results to")
    args, _ = parser.parse_known_args()
    main(**vars(args))