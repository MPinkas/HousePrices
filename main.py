import os
import pandas as pd
import numpy as np
import tensorflow as tf


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def load_data():
    dir = os.getcwd()
    path = "\\data\\train.csv"
    path2 = "\\data\\data.txt"

    df = pd.read_csv(dir + path)

    X = df.to_numpy()
    X = X[:, 1:-1]  # remove labels from training data and index of row (which is just position + 1)
    y = X[:, -1]  # labels for training data

    clean_lot_data(X)

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

def grade_index(X: np.array, index: int, grading: np.array, nan_data=False):

    data = X[:, index]
    graded = np.zeros(shape=data.shape)

    for i in range(grading.shape[0]):
        search = np.where(data == grading[i])[0]
        graded[search] = i

    return graded, grading



def clean_lot_data(X: np.array):

    zzz = X[:, 0]
    to_one_hot = [0, 1, 4, 7, 9, 11, 12, 13, 14, 15, 20, 21, 22, 23, 28, 38, 64, 77, 78]
    to_one_hot_nan = [5, 24, 41, 57, 59, 73]

    no_adjustnebts = [2, 3, 8, 10, 16, 17, 18, 19, 25, 33, 35, 36, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 60,
                      61, 65, 66, 67, 68, 69, 70, 74, 75]

    graded_index = [6, 10, 26, 27, 39, 40, 52, 54]
    graded_index_nan = [29, 30, 31, 32, 34, 56, 62, 63, 71, 72]

    to_adjust = [8, 18, 19, 58, 76]

    not_needed = [37]

    # CONSIDER - adjust features like exeterion1st and 2nd (that go together) so that they can be one hotted together
    # ADD - function that receives data with textual scale and convert it to number grades
    # check how no garage (index 57) interacts with garage year built (58), verify garage size is 0 (59, 60)

    zzz = X[:, 7]

    '''
    for i in range(78):
        try:
            zzz = np.unique(X[:, i])
        except TypeError:
            print(i)
    '''

    a, b = grade_index(X, 6, np.array(['Reg', 'IR1', 'IR2', 'IR3']))

    yyy = 0





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
