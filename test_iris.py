from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from model_class import *


def iris_data():
    """
    return x in n*p ndarray, y in  n*1 ndarray
    """
    iris = load_iris()
    return iris.data, iris.target


def normalize(x):
    mean = np.mean(x, axis=0)  # a vector of means for different predict variables
    std = np.std(x, axis=0)

    for i in range(len(x[0])):
        x[:,i] = (x[:,i]-mean[i])/std[i]

def test_iris():
    X, y = iris_data()

    #  To train our data, first we need to do z-score normalization and shuffle
    normalize(X) # here we can perform np.mean and np.std by vector

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=8) # notice the
    # order of the return!

    model = Linear_mylab()
    B = model.fit(x_train, y_train)
    print(B)
