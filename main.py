import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Notebook
import io
import sys
import ai_model
from ai_model import BACKGROUND_HEX, PLANET_PATH_KEY, TRAINING_PATH_KEY
from PIL import Image, ImageTk
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.preprocessing.image_dataset import ALLOWLIST_FORMATS


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


def load_recognition_gui(tab_layout, settings):
    """Recognition Tab GUI"""
    tab_recognition = tk.Frame(tab_layout)
    tab_recognition.pack(fill='both')
    label = tk.Label(tab_recognition, text="Recognition")
    label.pack()
    tab_recognition.pack(fill='both')
    return tab_recognition


def load_training_gui(tab_layout, settings):
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

        @ai_model.run_in_thread
        def find_classes():
            # shows expected class_name, this may be blocking heavily
            _, _, class_names = dataset_utils.index_directory(planet_path, 'inferred', ALLOWLIST_FORMATS)
            print("Classes found:", ", ".join(class_names))
        find_classes()
        settings.update(ai_model.save_key(PLANET_PATH_KEY, planet_path))

    # When save folder button is clicked
    def save_path_folder():
        nonlocal save_path
        save_path = filedialog.askdirectory(initialdir=save_path)
        print("Save Path:", save_path)
        settings.update(ai_model.save_key(TRAINING_PATH_KEY, save_path))

    # When running folder button is clicked
    def running_train():
        if planet_path is None:
            print("Planets path folder is not selected. Please select a path.")
        elif save_path is None:
            print("Saving Path is not selected. Please select a path.")
        else:
            ai_model.data_train(planet_path)

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


def load_gui(options):
    window = tk.Tk()
    window.title("Planet Recognition")

    # Tab Layout
    tab_frame = tk.Frame(window, bg=BACKGROUND_HEX)
    tab_frame.pack(fill='both')
    tab_layout = Notebook(tab_frame)

    # Create tabs
    tab_recognition = load_recognition_gui(tab_layout, options)
    tab_training = load_training_gui(tab_layout, options)

    # Add the created tab
    tab_layout.add(tab_recognition, text="Recognition")
    tab_layout.add(tab_training, text="AI Training")

    # Starts
    tab_layout.pack(fill='both')
    window.mainloop()


if __name__ == '__main__':
    loaded_settings = ai_model.load_settings()
    load_gui(loaded_settings)
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
