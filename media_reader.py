"""
Method for reading images and video from
"""

import os
import pandas as pd

from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def read_images(path) -> pd.DataFrame:
    """ Read an image and return to main """
    categories = os.listdir(path)
    in_arr = []
    out_arr = []

    # Get rid of ds store folder
    try:
        categories.remove('.DS_Store')
    except ValueError:
        pass

    # Load the images and create separate data and label arrays
    for i in categories:
        if i[0] == '.':
            pass
        else:
            print(f':Loading... species: {i}')
            img_path = os.path.join(path, i)
            imgs = os.listdir(img_path)

            # Get rid of any possible ds store folders in the images list
            try:
                imgs.remove('.DS_Store')
            except ValueError:
                pass

            for img in imgs:
                img_array = imread(os.path.join(img_path, img))
                img_resized = resize(img_array, (150, 150, 3))  # Standardization step should be separated maybe
                in_arr.append(img_resized.flatten())
                out_arr.append(categories.index(i))

            print(f'Loaded species:{i} successfully')

    # Flatten the data
    flat_data = np.array(in_arr)
    target = np.array(out_arr)

    # Insert into dataframe
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    return df


def read_video_file():
    """ For testing and validation. Read a video from a file for ml validation """


def read_video():
    """ Read video stream from a device and return to main """
