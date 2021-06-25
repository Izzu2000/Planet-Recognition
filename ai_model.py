import threading
import numpy as np
import json
import contextlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Constants
SETTINGS_PATH = 'settings.json'
PLANET_PATH_KEY = "planet_folder_path"
TRAINING_PATH_KEY = 'training_history_path'
RECOGNITION_PATH_KEY = 'recognition_history_path'
CLASS_KEY = 'classes_saved'
BACKGROUND_HEX = '#acacac'
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


def run_in_thread(func):
    """Run the decorated function in a thread, and directly run when blocking is set to True"""
    def runner(*args, blocking=False, **kwargs):
        if not blocking:
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
            return thread
        return func(*args, **kwargs)
    return runner


def create_model(objects, settings):
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
    if settings.get(TRAINING_PATH_KEY) is None:
        path = 'training_history'
        settings.update(save_key(TRAINING_PATH_KEY, path))
    return model


@run_in_thread
def data_train(planet_path, settings, epochs=200):
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

    settings.update(save_key(CLASS_KEY, train_ds.class_names))

    print("Identified Classes:", ", ".join(train_ds.class_names))
    # Normalization
    # changes from [0, 255] into [0, 1] ranges,
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    def func(x, y):
        print("Normalizing")
        return normalization_layer(x), y

    train_ds.map(func)

    # Create model
    model = create_model(len(train_ds.class_names), settings)

    # trains it with whatever epoch given
    model.fit(train_ds, validation_data=valid_ds, epochs=epochs)
    print("Training ended...")
    path = settings.get(TRAINING_PATH_KEY) or "current directory"
    print("Saving data to", path)
    # saves the model
    model.save(path)


def predict_data(path, label, data):
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
    model = keras.models.load_model(data.get(RECOGNITION_PATH_KEY))
    predictions = model.predict(img_array)
    # returns a list of scores for all 5 nodes
    score = tf.nn.softmax(predictions[0])

    # get all class as a list
    classes = data[CLASS_KEY]
    # np.argmax gets the index of the highest value
    # np.max gets the highest value of the list
    predicted = classes[np.argmax(score)]
    score = 100 * np.max(score)
    return predicted, score, label


def predict_single_data(path, settings):
    path = path.replace("/", "\\")
    x = path.split('\\')[-1]
    y = x.split()[0]
    z = y.split('.')[0]
    return predict_data(path, z.capitalize(), settings)
