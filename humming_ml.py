"""
ML trainer and inference file for video and photo
"""
import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class HummingMLPhoto:
    def __init__(self, model_name, load=False):
        self.model_name = model_name
        self.model = None
        self.class_names = None

        if load:
            self.load_model()

    def train_photo(self, path, plot=False):
        """ Method to train the hummingbird model """
        # Some default parameters
        batch_size = 32
        img_height = 180
        img_width = 180

        # Call to read images
        path = pathlib.Path(path).with_suffix('')

        # training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        # Validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        # The category names
        class_names = train_ds.class_names

        # Set the data to stay in memory
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Image augmentation
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=(img_height,
                                               img_width,
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        # Define our model
        num_classes = len(class_names)

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),  # Standardization layer
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model
        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        # Show the results if wanted
        if plot:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        self.model = model
        self.class_names = class_names

    def infer_photo(self, image_path):
        """ Method to evaluate an image or video stream """
        if self.model is None:
            print("No model has been loaded! Train one or load one from the models directory...")
            return

        img_height = 180
        img_width = 180

        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def image_clustering(self, ):
        """ Method to cluster images to verify the effectiveness of the standardization """

    def define_model_name(self, file_name):
        """ Method to change the file name for the model """
        self.model_name = file_name

    def save_model(self):
        """ Method to save the trained model """
        if 'photo' in self.model_name:
            self.model.save(f'models/{self.model_name}.pkl')
        else:
            self.model.save(f'models/{self.model_name}_photo.pkl')

    def load_model(self):
        """ Method to load the model from a file """
        try:
            self.model = tf.keras.models.load_model(f'models/{self.model_name}.pkl')
        except FileNotFoundError:
            print("Model of that name doesn't exist. Verify file name and try again.")


class HummingMLVideo:
    def __init__(self, model_name, load=False):
        self.model_name = model_name
        self.model = None
        self.class_names = None

        if load:
            self.load_model()

    def train_model(self, path):
        """ Method for training a video model """

    def save_model(self):
        """ Method to save the trained model """
        self.model.save(f'models/{self.model_name}.pkl')

    def load_model(self):
        """ Method to load the model from a file """
        self.model = tf.keras.models.load_model(f'models/{self.model_name}.pkl')

