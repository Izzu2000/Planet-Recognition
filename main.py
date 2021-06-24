import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Notebook
import io
import sys
import ai_model
from ai_model import BACKGROUND_HEX, PLANET_PATH_KEY, TRAINING_PATH_KEY
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


def fit_center_calculation(canvas, img_loaded, maxw, maxh):
    maxw
    w, h = img_loaded.size
    wscale = maxw / w
    hscale = maxh / h
    cw, ch = maxw / 2, maxh / 2
    scale = min(wscale, hscale)
    size = [int(x * scale) for x in (w, h)]
    resized = img_loaded.resize(size)
    photo = ImageTk.PhotoImage(resized)
    canvas.create_image(cw, ch, image=photo, anchor='center')
    canvas.image = photo


def load_recognition_gui(tab_layout, settings):
    """Recognition Tab GUI"""
    padding = {'padx': 7, 'pady': 7}
    tab_recognition = tk.Frame(tab_layout, bg=BACKGROUND_HEX)
    save_path = settings.get(TRAINING_PATH_KEY)

    # When save folder button is clicked
    def save_path_folder():
        nonlocal save_path
        save_path = filedialog.askdirectory(initialdir=save_path)
        print("Save Path:", save_path)
        settings.update(ai_model.save_key(TRAINING_PATH_KEY, save_path))

    maxw, maxh = 620, 400

    @ai_model.run_in_thread
    def recognise_path():
        nonlocal canvas
        recog_path = filedialog.askopenfile()
        pg.start()
        with Image.open(recog_path.name) as img_loaded:
            canvas.config(bg='black')
            fit_center_calculation(canvas, img_loaded, maxw, maxh)
        predicted, score, _ = ai_model.test_single_data(recog_path.name, settings)
        actual_prediction["text"] = predicted
        actual_confidence["text"] = f"{score:.2f}%"
        pg.stop()

    predict_box = tk.Frame(tab_recognition, bg="#707070")
    label_prediction = tk.Label(predict_box, text="Prediction", font=('Arial', 13, 'bold'), fg="#ffffff", bg="#707070")
    label_prediction.grid(row=0, column=0, sticky='nw', padx=7, pady=7)
    actual_prediction = tk.Label(predict_box, text="NA", font=('Arial', 25, 'bold'), fg="#04fd99", bg="#707070")
    actual_prediction.grid(row=1, column=0, padx=7, pady=7)
    predict_box.grid(row=1, column=3, sticky="ew", **padding)

    confidence_box = tk.Frame(tab_recognition, bg="#707070")
    label_confidence = tk.Label(confidence_box, text="Confidence Level", font=('Arial', 14, 'bold'), fg="#ffffff", bg="#707070")
    label_confidence.grid(row=0, column=0, padx=7, pady=7)
    actual_confidence = tk.Label(confidence_box, text="NA", font=('Arial', 25, 'bold'), fg="#ffffff", bg="#707070")
    actual_confidence.grid(row=2, column=0, padx=7, pady=7, sticky='nw')
    confidence_box.grid(row=2, column=3, sticky="ew", **padding)

    pg = tk.ttk.Progressbar(tab_recognition)
    pg.grid(row=3, column=3, sticky="ew", **padding)

    canvas = tk.Canvas(tab_recognition, width=maxw, height=maxh)
    with Image.open('Resources/Default_image.png') as img_loaded:
        fit_center_calculation(canvas, img_loaded, maxw, maxh)
    canvas.grid(row=1, column=0, rowspan=4, columnspan=3, **padding)
    recog_but = create_button_image(
        tab_recognition,
        image=r'Resources\Folder_Icon.png',
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
