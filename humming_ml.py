"""
ML trainer and inference file
"""

import joblib
import media_reader as mr
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train(path) -> [list, list, GridSearchCV]:
    """ Method to train the hummingbird model """
    # Call to read images
    _, ml_data = mr.read_images(path)

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

    # Print the accuracy of the model
    print(f"The model is {accuracy * 100}% accurate")

    return y_test, y_pred, model


def inference(model, path) -> None:
    """ Method to evaluate an image or video stream """
    labels, ml_data = mr.read_images(path)

    probability = model.predict_proba(ml_data[0])
    for ind, val in enumerate(labels):
        print(f'{val} = {probability[0][ind] * 100}%')
    print("The predicted image is : " + labels[model.predict(ml_data[0])[0]])


def image_clustering():
    """ Method to cluster images to verify the effectiveness of the standardization """


def save_model(model) -> None:
    """ Method to save the trained model """
    joblib.dump(model, 'models/hummifier.pkl')


def load_model() -> GridSearchCV:
    """ Method to load the model from a file """
    return joblib.load('models/hummifier.pkl')


def print_classification_report(pred, test, labels) -> True:
    """ Print a classification report """
    print(classification_report(test, pred, target_names=labels))
