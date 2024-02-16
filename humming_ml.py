"""
ML trainer and inference file
"""

import media_reader as mr
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def train(path) -> [list, list, GridSearchCV]:
    """ Method to train the hummingbird model """
    # Call to read images
    ml_data = mr.read_images(path)

    # image data
    x = ml_data.iloc[:, :-1]

    # labels
    y = ml_data.iloc[:, -1]

    # 80:20 split on train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

    # Defining the parameters grid for GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.0001, 0.001, 0.1, 1],
                  'kernel': ['rbf', 'poly']}

    # Creating a support vector classifier
    svc = svm.SVC(probability=True)

    # Creating a model using GridSearchCV with the parameters grid
    model = GridSearchCV(svc, param_grid)

    # Training the model using the training data
    model.fit(x_train, y_train)

    # Testing the model using the testing data
    y_pred = model.predict(x_test)

    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_pred, y_test)

    # Print the accuracy of the model TODO Change to logging
    print(f"The model is {accuracy * 100}% accurate")

    return y_test, y_pred, model


def inference(model):
    """ Method to evaluate an image or video stream """


def image_clustering():
    """ Method to cluster images to verify the effectiveness of the standardization """


def print_classification_report(pred, test, labels):
    """ Print a classification report """
    print(classification_report(test, pred, target_names=labels))
