import cv2
import numpy as np
import glob
import skimage.feature as feature
import pandas as pd
import os
from pathlib import Path

from PIL import Image, ImageTk
import tkinter as tk

window = tk.Tk()

def display_images(original_img, grayscale_img):
    # Convert images to PIL format
    original_pil = Image.fromarray(original_img)
    grayscale_pil = Image.fromarray(grayscale_img)

    # Create Tkinter image objects
    original_tk = ImageTk.PhotoImage(original_pil)
    grayscale_tk = ImageTk.PhotoImage(grayscale_pil)

    # Create labels to display images
    original_label = tk.Label(window, text="Original", padx=10, pady=10)
    grayscale_label = tk.Label(window, text="Grayscale", padx=10, pady=10)
    original_image_label = tk.Label(window, image=original_tk)
    grayscale_image_label = tk.Label(window, image=grayscale_tk)

    # Add labels to the window
    original_label.grid(row=0, column=0)
    grayscale_label.grid(row=0, column=1)
    original_image_label.grid(row=1, column=0)
    grayscale_image_label.grid(row=1, column=1)

    # Update the Tkinter window
    window.update()

    # Wait for any key press to close the window
    window.mainloop()

def getData(pathDir):
    path = glob.glob(pathDir)
    filename = []
    images = []
    contrast = []
    correlation = []
    homogeneity = []
    energy = []
    i = 0

    for imagepath in path:
        image_spot = cv2.imread(imagepath,cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        name = Path(imagepath).stem
        filename.append(name)
        images.append(gray)

        # display_images(image_spot, gray)

        # print(i)
        i += 1
        if i == 1199:
            break

    for item in images:
        graycom = feature.graycomatrix(item, [1], [3*np.pi/4], levels=256)
        contrast.append(feature.graycoprops(graycom, 'contrast').item())
        correlation.append(feature.graycoprops(graycom, 'correlation').item())
        homogeneity.append(feature.graycoprops(graycom, 'homogeneity').item())
        energy.append(feature.graycoprops(graycom, 'energy').item())

    return filename, contrast, correlation, homogeneity, energy

def getDataFrame(path):
    filename, contrast, correlation, homogeneity, energy = getData(path)
    header = ['filename', 'contrast', 'correlation', 'homogeneity', 'energy']
    df = pd.DataFrame(list(zip(filename, contrast, correlation, homogeneity, energy)), columns=header)
    df = df.assign(label=os.path.basename(os.path.dirname(path)))

    return df

if __name__ == "__main__":
    print(getData('../Data/cell_images/dummy/test/C170P131ThinF_IMG_20151119_120150_cell_49_U.png'))
