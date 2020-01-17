import argparse
import os
import pdb
import sys

import keras
import tensorflow

from dataset.dataset import Dataset
from model.models import Models
from model.metrics import Metrics
from model.model import Model
from util.logger import Logger
from util.datetime import timestamp


def main(dataset, model, dropout, bias_init, learning_rate, class_weights, metrics, epochs, save_path, log_path):
    args = locals()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger = Logger(__name__)
    fd = open(log_path, "a")
    old_fd = sys.stdout
    sys.stdout = fd
    logger.logger.info("Begin")
    print(" ".join(["--{} {}".format(key, str(val) if not isinstance(val, list) else " ".join(map(str, val))) 
          for key, val in args.items()]))

    dataset = Dataset(dataset)
    dataset.build()

    model = Model(model, dropout, bias_init, class_weights, learning_rate, metrics)
    model.build()
    
    model.train(dataset.train_gen, dataset.val_gen, epochs, save_path)

    logger.logger.info("End")
    sys.stdout = old_fd
    fd.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--model", type=Models, choices=list(Models), default=Models.RESNET50, help="Which base model to use")
    parser.add_argument("--dropout", type=float, help="Dropout rate (leave blank for no dropout)")
    parser.add_argument("--bias_init", type=float, help="Bias initializer to use for the last layer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate to use")
    parser.add_argument("--class_weights", type=float, nargs=2, help="Class weights to use for training")
    parser.add_argument("--metrics", type=Metrics, choices=list(Metrics), nargs="*", help="Which metrics to use for evaluation (F1 is on by default)")
    parser.add_argument("--epochs", type=int, default=2**64, help="Number of epochs to train for.")
    parser.add_argument("--save_path", type=str, default="./output/{}/model-{}.hdf5".format(timestamp(), "{epoch:03d}"), help="Path to save models to")
    parser.add_argument("--log_path", type=str, default="./output/{}/log.txt".format(timestamp()), help="Path to save logs to")
    args, _ = parser.parse_known_args()
    main(**vars(args))