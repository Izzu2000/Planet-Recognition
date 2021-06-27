import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
from tkinter.ttk import Notebook
import time
import io
import sys
import ai_model
from ai_model import BACKGROUND_HEX, PLANET_PATH_KEY, TRAINING_PATH_KEY, RECOGNITION_PATH_KEY
from collections import namedtuple
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


ImageButton = namedtuple("ImageButton", "button place")


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
    button.pack(fill="both")
    return ImageButton(button, button_border)


def new_path(button, path):
    text = button['text'].splitlines()
    if len(text) < 3:
        text.append(path)
    else:
        text[2] = path

    button['text'] = "\n".join(text)


def load_recognition_gui(tab_layout, settings):
    """Recognition Tab GUI"""
    padding = {'padx': 7, 'pady': 7}
    tab_recognition = tk.Frame(tab_layout, bg=BACKGROUND_HEX)
    tab_recognition.grid_columnconfigure(0, weight=1)

    # When save folder button is clicked
    def save_path_folder():
        nonlocal save_path
        if not (asked := filedialog.askdirectory(initialdir=save_path, title="Select AI Model Folder")):
            return
        save_path = asked
        print("Save Path:", save_path)
        new_path(save_but.button, save_path)
        settings.update(ai_model.save_key(RECOGNITION_PATH_KEY, save_path))
        recog_but.button['state'] = 'normal'
    max_size = 620, 400

    def fit_center_calculation(img):
        """Put the loaded image into the canvas at fit it to the center"""
        nonlocal canvas
        old_size = img.size
        scales = [x / y for x, y in zip(max_size, old_size)]
        middle_position = [x / 2 for x in max_size]
        scale = min(scales)
        size = [int(x * scale) for x in old_size]
        resized = img.resize(size)
        photo = ImageTk.PhotoImage(resized)
        canvas.create_image(*middle_position, image=photo, anchor='center')
        canvas.image = photo

    @ai_model.run_in_thread
    def recognise_path():
        """Function is triggered when a user click on recognition button"""
        nonlocal canvas
        started = True
        if (recog_path := filedialog.askopenfile(title="Select Planet Image")) is None:
            return

        @ai_model.run_in_thread
        def text_change():
            while True:
                for x in range(3):
                    if not started:
                        return
                    text = 'Predicting' + ("." * (x + 1))
                    actual_prediction["text"] = text
                    actual_confidence["text"] = text
                    time.sleep(1)

        recog_but.button['state'] = 'disabled'
        text_change()
        new_path(recog_but.button, recog_path.name)
        pg.start()
        try:
            with Image.open(recog_path.name) as img_loaded:
                canvas.config(bg='black')
                fit_center_calculation(img_loaded)
            # Where actual prediction starts
            predicted, score, _ = ai_model.predict_single_data(recog_path.name, settings)
            actual_prediction["text"] = predicted
            actual_confidence["text"] = f"{score:.2f}%"
        except Exception as e:
            # When error, show an error message and stop the prediction
            tkinter.messagebox.showerror(title="Error on prediction.", message=str(e))
            actual_prediction["text"] = "NA"
            actual_confidence["text"] = "NA"

        started = False
        pg.stop()
        recog_but.button['state'] = 'normal'

    FRAME_BACKGROUND = "#707070"
    
    title_frame = dict(font=('Arial', 13, 'bold'), fg="#ffffff", bg=FRAME_BACKGROUND)
    desc_text = dict(font=('Arial', 25, 'bold'), bg=FRAME_BACKGROUND, text="NA")
    predict_box = tk.Frame(tab_recognition, bg=FRAME_BACKGROUND)
    label_prediction = tk.Label(predict_box, text="Prediction", **title_frame)
    label_prediction.grid(row=0, column=0, sticky='nw', **padding)
    actual_prediction = tk.Label(predict_box, fg="#04fd99", **desc_text)
    actual_prediction.grid(row=1, column=0, sticky='nw', **padding)
    predict_box.grid(row=1, column=3, sticky="ew", **padding)

    confidence_box = tk.Frame(tab_recognition, bg=FRAME_BACKGROUND)
    label_confidence = tk.Label(confidence_box, text="Confidence Level", **title_frame)
    label_confidence.grid(row=0, column=0, sticky='nw', **padding)
    actual_confidence = tk.Label(confidence_box, fg="#ffffff", **desc_text)
    actual_confidence.grid(row=2, column=0, sticky='nw', **padding)
    confidence_box.grid(row=2, column=3, sticky="ew", **padding)

    pg = tk.ttk.Progressbar(tab_recognition)
    pg.grid(row=3, column=3, sticky="ew", **padding)

    w, h = max_size
    canvas = tk.Canvas(tab_recognition, width=w, height=h)
    with Image.open('Resources/Default_image.png') as img_loaded:
        fit_center_calculation(img_loaded)
    canvas.grid(row=1, column=0, rowspan=4, columnspan=3, **padding)
    recog_but = create_button_image(
        tab_recognition,
        image=r'Resources\Image_icon.png',
        title="Select Image",
        description="Select an image path that you want to ",
        bg='#dfdfdf',
        command=recognise_path
    )

    save_but = create_button_image(
        tab_recognition,
        image=r'Resources\Folder_Icon.png',
        title="Select AI Model",
        description="Path to the saved AI model.",
        bg='#dfdfdf',
        command=save_path_folder
    )
    recog_but.button['state'] = 'disabled'
    if save_path := settings.get(RECOGNITION_PATH_KEY):
        new_path(save_but.button, save_path)
        recog_but.button['state'] = 'normal'
    recog_but.place.grid(row=0, column=0, columnspan=2, sticky="ew", **padding)
    save_but.place.grid(row=0, column=2, columnspan=2, sticky="ew", **padding)
    tab_recognition.pack(fill='both')
    return tab_recognition


def load_training_gui(tab_layout, settings):
    """Training Tab GUI"""
    padding = {'padx': 7, 'pady': 7}
    tab_training = tk.Frame(tab_layout, bg=BACKGROUND_HEX)
    tab_training.grid_columnconfigure(0, weight=1)
    tab_training.grid_rowconfigure(0, weight=1)

    text_box = tk.Text(tab_training)
    sys.stdout = ConsoleWritter(text_box)

    # When training folder button is clicked
    def training_folder():
        nonlocal planet_path
        if not (asked_path := filedialog.askdirectory(initialdir=planet_path, title="Select Planet Images Folder")):
            return
        planet_path = asked_path
        new_path(planet_but.button, asked_path)
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
        if not (asked := filedialog.askdirectory(initialdir=save_path, title="Select Folder to Save Model")):
            return
        new_path(save_but.button, asked)
        save_path = asked
        print("Save Path:", save_path)
        settings.update(ai_model.save_key(TRAINING_PATH_KEY, save_path))

    # When running folder button is clicked
    @ai_model.run_in_thread
    def running_train():
        if planet_path is None:
            print("Planets path folder is not selected. Please select a path.")
        elif save_path is None:
            print("Saving Path is not selected. Please select a path.")
        else:
            start_but.button["state"] = "disabled"
            ai_model.data_train(planet_path, settings, epochs=2, blocking=True)
            start_but.button["state"] = "normal"

    # Create buttons
    planet_but = create_button_image(
        tab_training,
        image=r'Resources\Folder_Icon.png',
        title="Planets",
        description="Folder that contains all images for training.",
        bg='#dfdfdf',
        command=training_folder
    )
    planet_but.place.grid(row=0, column=1, columnspan=1, sticky='nesw', **padding)

    save_but = create_button_image(
        tab_training,
        image=r'Resources\Folder_Icon.png',
        title="Save Model",
        description="Path to save the AI model.",
        bg='#dfdfdf',
        command=save_path_folder
    )
    if planet_path := settings.get(PLANET_PATH_KEY):
        print("Training Path:", planet_path)
        new_path(planet_but.button, planet_path)
    if save_path := settings.get(TRAINING_PATH_KEY):
        print("Save Path:", save_path)
        new_path(save_but.button, save_path)
    save_but.place.grid(row=0, column=0, columnspan=1, sticky='nesw', **padding)
    text_box.grid(row=1, column=0, columnspan=2, sticky='nesw', **padding)
    start_but = create_button_image(
        tab_training,
        text_size=10,
        image=r'Resources\Start_icon.png',
        title="Start Training",
        bg='#dfdfdf',
        command=running_train
    )
    start_but.place.grid(row=2, column=1, sticky='e', **padding)
    tab_training.pack(fill='both')
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
