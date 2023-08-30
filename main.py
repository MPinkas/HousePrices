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


def basement_data(X: np.array):
    """
    cleans columns 32-36 which correspond to data about the size and quality of the basement
    :param X: training data before cleaning
    :return: (X.shape[0], 6) np array where every row gives the footage of a certain quality of basement
        (no basemet is a 0 row)
    """
    basement_finish = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf']
    basement_indexes = [32, 33, 34, 35, 36]
    data = X[:, basement_indexes]
    samples = X.shape[0]

    bsmnt_data = np.zeros(shape=(samples, len(basement_finish)))

    for i in range(samples):

        for j in range(len(basement_finish)):

            if basement_finish[j] == data[i][0]:
                bsmnt_data[i][j] = data[i][1]  # add quantity of type 1

            if basement_finish[j] == data[i][2]:
                bsmnt_data[i][j] = data[i][3]  # add quantity of type 2

        bsmnt_data[i][5] = data[i][4]  # add unfinished quantity

    return bsmnt_data


def fireplace_data(X: np.array):
    '''
    cleans the data about the number and quality of fireplaces
    :param X: training data before cleaning
    :return: (X.shape[0], 2) np array where every row is given by the number of fireplaces of every quality
    (in reality its a vector of all 0 except at most 1 coordiante)
    '''
    fp_quality = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    fp_indexes = [55, 56]
    data = X[:, fp_indexes]
    samples = X.shape[0]

    fp_data = np.zeros(shape=(samples, len(fp_quality)))

    for i in range(samples):

        for j in range(len(fp_quality)):

            if fp_quality[j] == data[i][1]:
                fp_data[i][j] = data[i][0]  # add quantity of type 1

    return fp_data


def clean_lot_data(X: np.array):

    zzz = X[:, 0]
    to_one_hot = [0, 1, 4, 7, 9, 11, 12, 13, 14, 15, 20, 21, 22, 23, 28, 38, 77, 78]
    to_one_hot_nan = [5, 24, 41, 57, 73]

    no_adjustments = [2, 3, 8, 10, 16, 17, 18, 19, 25, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 60,
                      61, 65, 66, 67, 68, 69, 70, 74, 75]

    graded_index = [6, 10, 26, 27, 39, 40, 52, 54, 64]
    graded_scale = []
    graded_scale.append(np.array(['Reg', 'IR1', 'IR2', 'IR3']))
    graded_scale.append(np.array(['Gtl', 'Mod', 'Sev']))
    graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    graded_scale.append(np.array(['N', 'Y']))
    graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    graded_scale.append(np.array(['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']))
    graded_scale.append(np.array(['N', 'P', 'Y']))

    graded_index_nan = [29, 30, 31, 59, 62, 63, 71, 72]

    basement_index = [32, 33, 34, 35, 36]

    fp_index = [55, 56]

    to_adjust = [8, 18, 19, 58, 76]

    not_needed = [37]

    # CONSIDER - adjust features like exeterion1st and 2nd (that go together) so that they can be one hotted together
    # ADD - function that receives data with textual scale and convert it to number grades
    # check how no garage (index 57) interacts with garage year built (58), verify garage size is 0 (59, 60)
    # ^ nan garage (57) means year built is also nan (58) so is finish (59) but size is 0 (60, 61)

    # data on basement - If there is one basement material it is listed as though the second type is unf (34)
    # unfinished data is only applied in (36) (i.e. if material is listed as unf footage of that metrial is 0)

    d = fireplace_data(X)
    zzz = X[:, [55, 56]]
    yyy = 0





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
