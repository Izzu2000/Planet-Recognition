import tkinter as tk
import itertools
from tkinter import filedialog
import json
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Constants
SETTINGS_PATH = 'settings.json'
PLANET_PATH_KEY = "planet_folder_path"
TRAINING_PATH_KEY = 'training_history_path'
CLASS_KEY = 'classes_saved'
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128


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


def get_folder_image(path):
    for file in os.listdir(path):
        full_path = f"{path}/{file}"
        print("Reading:", full_path)
        if not os.path.isdir(full_path):
            yield full_path
        else:
            yield (file, [*get_folder_image(full_path)])


def get_folder(path):
    return {k: v for k, v in get_folder_image(path)}


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

    get_files = list(get_folder(planet_path))
    text_box = tk.Text()
    text_box.pack()
    window.mainloop()


def process_path(file_path, label):
    image_string = tf.io.read_file(file_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    img = tf.image.resize(image_decoded, [IMAGE_HEIGHT, IMAGE_WIDTH])
    print(file_path, label)
    return img, label


def produce_dataset(dataset_dict):
    constant_label, constant_path = [], []
    for label, paths in dataset_dict.items():
        for path in paths:
            constant_label.append(label)
            constant_path.append(path)
    const_l = tf.constant(constant_label)
    const_p = tf.constant(constant_path)
    dataset = tf.data.Dataset.from_tensor_slices((const_p, const_l))
    dataset = dataset.map(process_path)
    dataset = dataset.batch(2)
    return dataset


def create_or_load_model(objects, **settings):
    if not (path := settings.get(TRAINING_PATH_KEY)):
        ACTIVATION_FUNC = 'relu'
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

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        path = 'training_history'
        settings.update(save_key(TRAINING_PATH_KEY, path))
        return model

    return keras.models.load_model(path)


def data_train(**settings):
    planet_path = settings.get(PLANET_PATH_KEY)
    BATCH_SIZE = 5
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        planet_path,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE)

    settings = save_key(CLASS_KEY, train_ds.class_names)

    print("classes:", train_ds.class_names)
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     planet_path,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    #     batch_size=BATCH_SIZE)

    # Normalization
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Get/create model
    model = create_or_load_model(len(train_ds.class_names))

    epochs = 50
    model.fit(train_ds, epochs=epochs)
    model.save(settings.get(TRAINING_PATH_KEY))


def test_data(path, label, **data):
    img = keras.preprocessing.image.load_img(
        path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    model = keras.models.load_model(history_path)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    classes = data[CLASS_KEY]
    print("Predicted Planet:", classes[np.argmax(score)], 100 * np.max(score))
    print("Actual Planet:", label)


if __name__ == '__main__':
    settings = load_settings()
    # load_GUI(**settings)
    # data_train(**settings)
    history_path = settings.get(TRAINING_PATH_KEY)
    root = 'C:/Users/izzu/PycharmProjects/Planet-Recognition/Test Data'
    for x in os.listdir(root):
        y = x.split()[0]
        z = y.split('.')[0]
        test_data(f"{root}/{x}", z.capitalize(), **settings)
        print()
