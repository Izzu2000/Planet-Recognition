import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Notebook
import threading
import json
import contextlib
import numpy as np
import os
import io
import tensorflow as tf
import sys
from PIL import Image, ImageTk
from tensorflow import keras
from tensorflow.keras import layers, models

# Constants
SETTINGS_PATH = 'settings.json'
PLANET_PATH_KEY = "planet_folder_path"
TRAINING_PATH_KEY = 'training_history_path'
CLASS_KEY = 'classes_saved'
BACKGROUND_HEX = '#acacac'
IMAGE_WIDTH, IMAGE_HEIGHT = 700, 700


class ConsoleWritter(io.IOBase):
    """This updates from console to the Text object"""
    def __init__(self, obj):
        self.obj = obj

    def write(self, arg):
        # Handle text everytime stdout receives a write call
        text = self.obj
        text.configure(state='normal')
        # \r indicates to update the text rather than adding new textline
        if arg and arg[0] == '\r':
            text.delete('current linestart', 'current lineend+1c')
            text.insert('end', '\n')
        text.insert('end', arg)
        text.see("end")
        text.configure(state='disabled')


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


def create_button_image(window, *, image, title, description=None, padx=7, text_size=11, **kwargs):
    font = ('Arial', text_size, 'bold')
    button_text = '\n'.join((title, description)) if description else title
    button_border = tk.Frame(window, highlightbackground="#858585",
                             highlightthickness=2, bd=0)
    button = tk.Button(
        button_border,
        text=button_text,
        anchor='w',
        justify='left',
        font=font,
        padx=padx,
        borderwidth=0,
        **kwargs
    )

    with Image.open(image).convert('RGBA') as img_loaded:
        img_loaded = img_loaded.resize((50, 50))
        photo = ImageTk.PhotoImage(img_loaded)
        button.image = photo
        button.config(image=photo, compound=tk.LEFT)
    button.pack()
    return button_border


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
        # Saving model
        path = 'training_history'
        settings.update(save_key(TRAINING_PATH_KEY, path))
        return model

    return keras.models.load_model(path)


def data_train(planet_path, epochs=200):
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

    print("Identified Classes:", ", ".join(train_ds.class_names))
    # Normalization
    # changes from [0, 255] into [0, 1] ranges,
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    def func(x, y):
        print("Normalizing")
        return normalization_layer(x), y

    train_ds.map(func)

    # Get/create model
    model = create_or_load_model(len(train_ds.class_names))

    # trains it with whatever epoch given
    model.fit(train_ds, validation_data=valid_ds, epochs=epochs)
    print("Training ended...")
    path = settings.get(TRAINING_PATH_KEY) or "current directory"
    print("Saving data to", path)
    # saves the model
    model.save(path)


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
    print("Predicted Planet:", predicted, score)
    print("Actual Planet:", label)
    return predicted, score, label


def get_training_folder(ask=False, **options):
    if not (planet_path := options.get(PLANET_PATH_KEY) and not ask):
        planet_path = filedialog.askdirectory()
        options.update({PLANET_PATH_KEY: planet_path})
        save(options)
    print("Training Path:", planet_path)
    return planet_path


def test_single_data(path):
    path = path.replace("/", "\\")
    x = path.split('\\')[-1]
    y = x.split()[0]
    z = y.split('.')[0]
    return test_data(path, z.capitalize(), **settings)


def test_folder_data(root):
    def predicting(x):
        predicted, _, label = test_single_data(f"{root}/{x}")
        print()
        return predicted == label

    results = [*map(predicting, os.listdir(root))]
    print("Final Accuracy:", sum(results) / len(results) * 100)


def load_recognition_GUI(tab_layout, **settings):
    """Recognition Tab GUI"""
    tab_recognition = tk.Frame(tab_layout)
    tab_recognition.pack(fill='both')
    label = tk.Label(tab_recognition, text="Recognition")
    label.pack()
    tab_recognition.pack(fill='both')
    return tab_recognition


def load_training_GUI(tab_layout, **settings):
    """Training Tab GUI"""
    padding = {'padx': 7, 'pady': 7}
    tab_training = tk.Frame(tab_layout, bg=BACKGROUND_HEX)
    tab_training.grid_columnconfigure(0, weight=1, uniform="fred")
    tab_training.pack(fill='both')

    text_box = tk.Text(tab_training)
    sys.stdout = ConsoleWritter(text_box)
    if planet_path := settings.get(PLANET_PATH_KEY):
        print("Training Path:", planet_path)
    if save_path := settings.get(TRAINING_PATH_KEY):
        print("Save Path:", save_path)

    # When training folder button is clicked
    def training_folder():
        nonlocal planet_path
        planet_path = filedialog.askdirectory(initialdir=planet_path)
        print("Training Path:", planet_path)
        settings.update(save_key(PLANET_PATH_KEY, planet_path))

    # When save folder button is clicked
    def save_path_folder():
        nonlocal save_path
        save_path = filedialog.askdirectory(initialdir=save_path)
        print("Save Path:", save_path)
        settings.update(save_key(TRAINING_PATH_KEY, save_path))

    # When running folder button is clicked
    def running_train():
        if planet_path is None:
            print("Planets path folder is not selected. Please select a path.")
        elif save_path is None:
            print("Saving Path is not selected. Please select a path.")
        else:
            thread = threading.Thread(target=data_train, args=(planet_path,))
            thread.start()

    # Create buttons
    place = create_button_image(
        tab_training,
        image=r'Resources\Folder_Icon.png',
        title="Planets",
        description="Folder that contains all images for training.",
        bg='#dfdfdf',
        command=training_folder
    )
    place.grid(row=1, column=0, **padding)

    place = create_button_image(
        tab_training,
        image=r'Resources\Folder_Icon.png',
        title="Save Model",
        description="Path to save the AI model.",
        bg='#dfdfdf',
        command=save_path_folder
    )
    place.grid(row=1, column=1, **padding)
    text_box.grid(row=2, column=0, columnspan=2, **padding)
    place = create_button_image(
        tab_training,
        text_size=10,
        image=r'Resources\Start_icon.png',
        title="Start Training",
        bg='#dfdfdf',
        command=running_train
    )
    place.grid(row=3, column=1, sticky='e', **padding)
    return tab_training


def load_GUI(**options):
    window = tk.Tk()
    window.title("Planet Recognition")

    # Tab Layout
    tab_frame = tk.Frame(window, bg=BACKGROUND_HEX)
    tab_frame.pack(fill='both')
    tab_layout = Notebook(tab_frame)

    # Create tabs
    tab_recognition = load_recognition_GUI(tab_layout, **options)
    tab_training = load_training_GUI(tab_layout, **options)

    # Add the created tab
    tab_layout.add(tab_recognition, text="Recognition")
    tab_layout.add(tab_training, text="AI Training")

    # Starts
    tab_layout.pack(fill='both')
    window.mainloop()


if __name__ == '__main__':
    settings = load_settings()
    load_GUI(**settings)
    # Asks for training path
    # training_path = r"C:\Users\sarah\PycharmProjects\Planet-Recognition\Planet"
    # # Train the model
    # data_train(r"C:\Users\sarah\PycharmProjects\Planet-Recognition\Planet", epochs=2)

    # # Do this if you wanna check every test data there are
    # root = r'C:\Users\sarah\PycharmProjects\Planet-Recognition\Test Images'
    # test_folder_data(root)
    #
    # # # Do this if you want to test 1 data
    # path = r"C:\Users\sarah\PycharmProjects\Planet-Recognition\Test Images\Earth 2.jpg"
    # test_single_data(path)
