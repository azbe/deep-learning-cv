import argparse
import os
import pdb

import keras
import tensorflow

from f1_score_callback import F1ScoreCallback


def main(dataset_path):
    model = keras.applications.ResNet50(include_top=False, weights=None)
    inputs, net = model.input, model.output
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.GlobalAveragePooling2D()(net)
    net = keras.layers.Dense(1, activation=keras.activations.sigmoid, bias_initializer=keras.initializers.Constant([-1.748545323]))(net)
    model = keras.models.Model(inputs=inputs, outputs=net)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ])
    
    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.,
        horizontal_flip=True,
        vertical_flip=True)
    train_gen = train_gen.flow_from_directory(
        os.path.join(dataset_path, "train"), 
        batch_size=32,
        target_size=(224, 224),
        class_mode="binary",
        shuffle=True)

    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    val_gen = val_gen.flow_from_directory(
        os.path.join(dataset_path, "validation"), 
        batch_size=32,
        target_size=(224, 224),
        class_mode="binary",
        shuffle=True)

    try:
        model.fit_generator(
            train_gen, 
            epochs=100000,
            verbose=1,
            callbacks=[
                F1ScoreCallback(val_gen, steps=125, log_path=os.path.join("models", "f1_log.txt")),
                keras.callbacks.ModelCheckpoint(os.path.join("models", "model-{epoch:03d}.hdf5"), save_best_only=False),
            ],
            validation_data=val_gen,
            validation_steps=125,
            class_weight={0: 0.587013456, 1: 3.373118838},)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data")
    args, _ = parser.parse_known_args()
    main(**vars(args))