import tkinter as tk
from tkinter import filedialog
import json
import contextlib
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Constants
SETTINGS_PATH = 'settings.json'
PLANET_PATH_KEY = "planet_folder_path"
TRAINING_PATH_KEY = 'training_history_path'
CLASS_KEY = 'classes_saved'
IMAGE_WIDTH, IMAGE_HEIGHT = 700, 700


def save(data):
    with open(SETTINGS_PATH, 'w+') as j:
        json.dump(data, j, indent=4)
    return data


def load():
    with open(SETTINGS_PATH, 'r') as j:
        return json.load(j)


def save_key(key, value):
    s = load()
    s.update({key: value})
    return save(s)


def load_settings():
    with contextlib.suppress(IOError):
        return load()
    return save({})


def load_GUI(**options):
    window = tk.Tk()
    top_frame = tk.Frame(master=window, width=20, height=50, bg="red")
    top_frame.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    inner_frame = tk.Frame(master=top_frame, height=40, bg='white')
    inner_frame.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    label1 = tk.Label(master=inner_frame, text="Root Planet Folder Path", bg="white")
    label1.place(x=5, y=5)
    inner_frame2 = tk.Frame(master=top_frame, bg='white')
    inner_frame2.pack(fill=tk.BOTH, side=tk.LEFT)

    planet_path_GUI = tk.Entry(master=inner_frame2, text="")
    planet_path_GUI.place(x=0, y=0)

    if not (planet_path := options.get(PLANET_PATH_KEY)):
        planet_path = filedialog.askdirectory()
        options.update({PLANET_PATH_KEY: planet_path})
        save(options)

    text_box = tk.Text()
    text_box.pack()
    window.mainloop()


def create_or_load_model(objects, **settings):
    if not (path := settings.get(TRAINING_PATH_KEY)):
        ACTIVATION_FUNC = 'relu'

        # Input shape is (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # - reduce computational time if image size is lower
        # - introduces the same resolution for all images input
        # 3 refers to the channels

        # Using max pooling
        # - to focus on the planet itself
        # - ignore the dark background of space

        # activation: relu
        # - to extract feature from planets

        # Flatten is the conversion to be inputted into the node

        # it contain 128 nodes
        # Final dense is for the 5 objects classified
        model = models.Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
            layers.Conv2D(16, 3, padding='same', activation=ACTIVATION_FUNC),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation=ACTIVATION_FUNC),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation=ACTIVATION_FUNC),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation=ACTIVATION_FUNC),
            layers.Dense(objects)
        ])

        # Adam is the best among the adaptive optimizers for most cases
        # -combinations of Adadelta and RMSprop, recommended to use adam for most cases

        # SparseCategoricalCrossentropy pretty much does computes the loss

        # metrics just shows our accuracy
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        path = 'training_history'
        settings.update(save_key(TRAINING_PATH_KEY, path))
        return model

    return keras.models.load_model(path)


def data_train(**settings):
    planet_path = settings.get(PLANET_PATH_KEY)
    BATCH_SIZE = 5
    # gets all the images from a root folder, Planet -> Jupiter, Mars, ...
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        planet_path,
        subset='training',
        seed=500,
        validation_split=0.1,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    )
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        planet_path,
        subset='validation',
        seed=500,
        validation_split=0.5,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    )

    settings = save_key(CLASS_KEY, train_ds.class_names)

    print("classes:", train_ds.class_names)
    # Normalization
    # changes from [0, 255] into [0, 1] ranges,
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    def func(x, y):
        print("Normalizing")
        return normalization_layer(x), y

    train_ds.map(func)

    # Get/create model
    model = create_or_load_model(len(train_ds.class_names))

    epochs = 200
    # trains it with whatever epoch given
    model.fit(train_ds, validation_data=valid_ds, epochs=epochs)

    # saves the model
    model.save(settings.get(TRAINING_PATH_KEY))


def test_data(path, label, **data):
    # Loads the image, and resize to the target_size
    img = keras.preprocessing.image.load_img(
        path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    # converts PIL image into numpy array
    # image -> [[r, g, b], ..., [r, g, b]] array
    img_array = keras.preprocessing.image.img_to_array(img)

    # [r, g, b] -> [a, r, g, b]
    img_array = tf.expand_dims(img_array, 0)

    # gets the saved model, predict and get the score
    model = keras.models.load_model(data.get(TRAINING_PATH_KEY))
    predictions = model.predict(img_array)
    # returns a list of scores for all 5 nodes
    score = tf.nn.softmax(predictions[0])

    # get all class as a list
    classes = data[CLASS_KEY]
    # np.argmax gets the index of the highest value
    # np.max gets the highest value of the list
    predicted = classes[np.argmax(score)]
    score = 100 * np.max(score)
    print("Predicted Planet:", predicted,score )
    print("Actual Planet:", label)
    return predicted, score


if __name__ == '__main__':
    settings = load_settings()
    # load_GUI(**settings) do this once gui is made
    # data_train(**settings)

    # Do this if you wanna check every test data there are
    root = 'C:/Users/izzu/PycharmProjects/Planet-Recognition/Test Images'


    def predicting(x):
        y = x.split()[0]
        z = y.split('.')[0]
        predicted, _ = test_data(f"{root}/{x}", z.capitalize(), **settings)
        print()
        return predicted == z.capitalize()

    results = [*map(predicting, os.listdir(root))]
    print("Final Accuracy:", sum(results) / len(results) * 100)
