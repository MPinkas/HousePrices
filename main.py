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

    if nan_data:
        #  clear out nan data for easier use
        for i in range(data.shape[0]):
            if pd.isnull(data[i]):
                data[i] = 'temp'

        tmp = np.array(['temp'])
        grading = np.concatenate([tmp, grading])

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
                fp_data[i][j] = data[i][0]
                break

    return fp_data


def garage_data(X: np.array):

    garage_indexes = [57, 58, 59, 60, 61, 62, 63]

    data = X[:, garage_indexes]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isnull(data[i][j]):
                if j == 1:
                    data[i][j] = 2024  # this input will yield age of -1 when there is no garage
                else:
                    data[i][j] = 'temp'

    types = ['2Types', 'Attchd', 'Basement', 'BuiltIn', 'CarPort', 'Detchd']


    garage_types = np.zeros(shape=(X.shape[0], len(types)))
    for i in range(X.shape[0]):
        for j in range(len(types)):

            if data[i][0] == types[j]:
                garage_types[i][j] = 1
                break

    age = 2023 - data[:, [1]].astype('int32')
    size = data[:, [3, 4]]

    fin, grd = grade_index(data, 2, np.array(['temp', 'Unf', 'RFn', 'Fin']), nan_data=False)
    qual, grd = grade_index(data, 5, np.array(['temp', 'Po', 'Fa', 'TA', 'Gd', 'Ex']), nan_data=False)
    cond, grd = grade_index(data, 6, np.array(['temp', 'Po', 'Fa', 'TA', 'Gd', 'Ex']), nan_data=False)

    t = np.column_stack((fin, qual, cond))

    return np.hstack((garage_types, age, size, t))


def utilities_data(X: np.array):

    # note here that training data contains 1 instance where it's not allpub (i.e. this might be garbage data)
    data = X[:, 8]
    utilities = np.ones(shape=(X.shape[0], 3))

    no_s = np.unique(data)
    no_s = np.where(data == 'NoSewr')[0]
    utilities[no_s, 2] = 0

    no_sw = np.where(data == 'NoSeWa')[0]
    utilities[no_sw, 1] = 0
    utilities[no_sw, 2] = 0

    no_swg = np.where(data == 'ELO')[0]
    utilities[no_swg, 0] = 0
    utilities[no_swg, 1] = 0
    utilities[no_swg, 2] = 0

    return utilities


def year_built_data(X: np.array):

    # note here that training data contains 1 instance where it's not allpub (i.e. this might be garbage data)
    data = X[:, [18, 19]]
    return 2023 - data

def year_sold_data(X: np.array):

    # note here that training data contains 1 instance where it's not allpub (i.e. this might be garbage data)
    data = X[:, [76]]
    return 2023 - data


def clean_lot_data(X: np.array):

    zzz = X[:, 0]
    to_one_hot = [0, 1, 4, 7, 9, 11, 12, 13, 14, 15, 20, 21, 22, 23, 28, 38, 77, 78]
    # ^ done
    to_one_hot_nan = [5, 24, 41, 73]
    # ^ done

    no_adjustments = [2, 3, 8, 10, 16, 17, 18, 19, 25, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53,
                      65, 66, 67, 68, 69, 70, 74, 75]
    # ^ done

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
    # ^ done

    nan_graded_index = [29, 30, 31, 71, 72]
    nan_graded_scale = []
    nan_graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    nan_graded_scale.append(np.array(['Po', 'Fa', 'TA', 'Gd', 'Ex']))
    nan_graded_scale.append(np.array(['No', 'Mn', 'Av', 'Gd']))
    nan_graded_scale.append(np.array(['Fa', 'TA', 'Gd', 'Ex']))
    nan_graded_scale.append(np.array(['MnWw', 'GdWo', 'MnPrv', 'GdPrv']))
    #  consider changing how nan valued columns are processed based on results later
    # add grading

    basement_index = [32, 33, 34, 35, 36]
    # ^ done

    fp_index = [55, 56]
    # ^ done

    garage_indexes = [57, 58, 59, 60, 61, 62, 63]
    # ^ done

    to_adjust = [18, 19, 76]
    # 8, 18, 19, 76 is done

    not_needed = [8, 37]

    # CONSIDER - adjust features like exeterion1st and 2nd (that go together) so that they can be one hotted together
    # check how no garage (index 57) interacts with garage year built (58), verify garage size is 0 (59, 60)
    # ^ nan garage (57) means year built is also nan (58) so is finish (59) but size is 0 (60, 61)

    one_hot = []
    data_adjustments = []

    for i in to_one_hot:
        col, items = one_hot_index(X, i)
        one_hot.append(col)
        data_adjustments.append(items)

    zzz1 = np.hstack(one_hot)
    # ^ this is how everthing needs to be glued together

    one_hot_nan = []
    for i in to_one_hot_nan:
        col, items = one_hot_index(X, i, True)
        one_hot_nan.append(col)
        data_adjustments.append(items)

    unadjusted = []
    for i in no_adjustments:

        unadjusted.append(X[:, i])
        data_adjustments.append(-1)

    graded_data = []
    for i in range(len(graded_index)):

        col, grading = grade_index(X, graded_index[i], graded_scale[i])
        graded_data.append(col)
        data_adjustments.append(grading)

    nan_graded_data = []
    for i in range(len(nan_graded_index)):
        col, grading = grade_index(X, nan_graded_index[i], nan_graded_scale[i], True)
        nan_graded_data.append(col)
        data_adjustments.append(grading)

    zzz = 0




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
