import os
import pandas as pd
import numpy as np
import tensorflow as tf


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def load_data():
    dir = os.getcwd()
    path = "\\data\\train.csv"

    df = pd.read_csv(dir + path)
    X = df.to_numpy()
    X = X[:, 1:-1]  # remove labels from training data and index of row (which is just position + 1)
    y = X[:, -1]  # labels for training data

    one_hot_index(X, 5)
    clean_lot_data(X)

    # features 3, 4
    f34 = X[:, 2:4]

    zzz = X[0]
    print(X[0])
    print(X[1])
    return X


def one_hot_index(X: np.array, index: int, nan_data=False):
    data = X[:, index]

    if nan_data:
        #  clear out nan data for easier use
        for i in range(data.shape[0]):
            if pd.isnull(data[i]):
                data[i] = 'temp'

    items = np.unique(data)

    new_data = np.zeros(shape=data.shape)
    for i in range(1, items.shape[0]):
        search = np.where(data == items[i])[0]
        new_data[search] = i

    depth = items.shape[0]

    one_hotted = tf.one_hot(indices=new_data, depth=depth).numpy()

    return one_hotted, items


def clean_lot_data(X: np.array):

    type_of_dwelling, tod_index = one_hot_index(X, 0)

    zoning, zoning_index = one_hot_index(X, 1)

    access, access_index = one_hot_index(X, 4)

    alley_access, alley_access_index = one_hot_index(X, 5, nan_data=True)

    # feature 7
    lot_shape = X[:, 6]

    new_lot_shape = np.zeros(shape=lot_shape.shape)
    shapes = ["Reg", "IR1", "IR2", "IR3"]

    for i in range(len(shapes)):
        search = np.where(lot_shape == shapes[i])[0]
        new_lot_shape[search] = i

    lot_shape = new_lot_shape


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
