import argparse
import os
import random


def main(dataset_path, train_imgs, random_seed):
    random.seed(random_seed)
    labels_file = os.path.join(dataset_path, "train_labels.txt")
    data_dir = os.path.join(dataset_path, "data", "data")
    train_dir = os.path.join(dataset_path, "train")
    validation_dir = os.path.join(dataset_path, "validation")
    test_dir = os.path.join(dataset_path, "test")

    labels = open(labels_file).readlines()
    labels = labels[1:]
    labels = [line.strip() for line in labels]
    labels = [(os.path.join(data_dir, "{}.png".format(line.split(",")[0])), int(line.split(",")[1])) for line in labels]
    random.shuffle(labels)
    train_labels = labels[:train_imgs]
    validation_labels = labels[train_imgs:]

    def makedir_and_subdirs(path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "0"), exist_ok=True)
        os.makedirs(os.path.join(path, "1"), exist_ok=True)
    makedir_and_subdirs(train_dir)
    makedir_and_subdirs(validation_dir)

    for path, label in train_labels:
        os.rename(path, os.path.join(train_dir, str(label), os.path.basename(path)))

    for path, label in validation_labels:
        os.rename(path, os.path.join(validation_dir, str(label), os.path.basename(path)))

    os.makedirs(test_dir, exist_ok=True)
    for path in os.listdir(data_dir):
        os.rename(os.path.join(data_dir, path), os.path.join(test_dir, os.path.basename(path)))

    os.removedirs(data_dir)
    os.remove(labels_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to dataset.")
    parser.add_argument("--train_imgs", type=int, default=15000 help="How many training files to keep in the training set.")
    parser.add_argument("--random_seed", type=int, default=30941, help="Random seed to use.")
    args, _ = parser.parse_known_args()
    main(**vars(args))