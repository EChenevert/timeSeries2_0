import pandas as pd
from scipy import stats
import numpy as np
import random

from sklearn import linear_model


def outlierrm(df, thres=3):
    """Dont put in long lats in here! Need Year and Site name lol"""
    df = df.dropna()  #.set_index(['Simple site', 'level_1'])
    # print(df.columns.values, len(df.columns.values), len(df))
    # switch = False
    # if 'Basins' in df.columns.values or 'Community' in df.columns.values:
    #     print('True')
    #     switch = True
    #     holdstrings = df[['Basins', 'Community']]
    #     df = df.drop(['Basins', 'Community'], axis=1)
    df = df.apply(pd.to_numeric)
    length = len(df.columns.values)
    for col in df.columns.values:
        df[col + "_z"] = stats.zscore(df[col])
    for col in df.columns.values[length:]:
        df = df[np.abs(df[col]) < thres]
    # print('length 1: ', len(df))
    df = df.drop(df.columns.values[length:], axis=1)
    # print('length 2: ', len(df))
    # if switch:
    #     df = pd.concat([df, holdstrings], join='inner', axis=1)
    return df


def cross_validation_split(dataset, folds: int):
    """
    https://www.kaggle.com/code/burhanykiyakoglu/k-nn-logistic-regression-k-fold-cv-from-scratch
    :param dataset:
    :param folds:
    :return:
    """
    dataset_split = []
    df_copy = dataset
    fold_size = int(df_copy.shape[0] / folds)

    # for loop to save each fold
    for i in range(folds):
        fold = []
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = random.randrange(df_copy.shape[0])
            # determine the index of this element
            index = df_copy.index[r]
            # save the randomly selected line
            fold.append(df_copy.loc[index].values.tolist())
            # delete the randomly selected line from
            # dataframe not to select again
            df_copy = df_copy.drop(index)
        # save the fold
        dataset_split.append(np.asarray(fold))

    return dataset_split


def kfoldCV(dataset, f=5, model="bayesian linear regression"):
    data = cross_validation_split(dataset, f)
    result = []
    # determine training and test sets
    for i in range(f):
        r = list(range(f))
        r.pop(i)
        for j in r:
            if j == r[0]:
                cv = data[j]
            else:
                cv = np.concatenate((cv, data[j]), axis=0)

        # apply the selected model
        # default is logistic regression
        if model == "bayesian linear regression":
            # default: alpha=0.1, num_iter=30000
            # if you change alpha or num_iter, adjust the below line
            br = linear_model.BayesianRidge()
            br.fit(cv[:, 0:4], cv[:, 4])
            ypred, ystd = br.predict(data[i][:, 0:4])


    return result

