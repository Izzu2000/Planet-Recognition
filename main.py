import tkinter as tk
from tkinter import filedialog
import json
import contextlib
import os

SETTINGS_PATH = 'settings.json'


def save(data):
    with open(SETTINGS_PATH, 'w+') as j:
        json.dump(data, j, indent=4)
    return data


def load_settings():
    with contextlib.suppress(IOError):
        with open(SETTINGS_PATH, 'r') as j:
            return json.load(j)

    return save({})


def get_folder_image(path):
    for file in os.listdir(path):
        full_path = f"{path}/{file}"
        print("Reading:", full_path)
        if os.path.isdir(full_path):
            yield f"folder: {file}"
            yield from get_folder_image(full_path)
        else:
            yield file


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

    PLANET_PATH_KEY = "planet_folder_path"
    if not (planet_path := options.get(PLANET_PATH_KEY)):
        planet_path = filedialog.askdirectory()
        options.update({PLANET_PATH_KEY: planet_path})
        save(options)

    get_files = list(get_folder_image(planet_path))
    text_box = tk.Text()
    text_box.pack()
    window.mainloop()


if __name__ == '__main__':
    settings = load_settings()
    load_GUI(**settings)
