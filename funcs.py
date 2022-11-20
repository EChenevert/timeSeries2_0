import pandas as pd
from scipy import stats
import numpy as np
from sklearn.utils import shuffle
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


# def kfold_split(dataset: pd.DataFrame, num_folds: int):
#     """
#     Will split the daatset into kfolds
#     """
#     foldsize = int(len(dataset)/num_folds)
#     # Just split the SHUFFLED df in order
#     phi_ls = []
#     for i in range(num_folds):
#         phi_ls.append(dataset.iloc[foldsize*i:foldsize*i + foldsize, :])
#     return phi_ls
#
#
# def repeated_fivefold_cross_validation(dataset: pd.DataFrame, num_folds: int, repeats: int,
#                                        model: linear_model.BayesianRidge, target_column: str):
#     """
#     """
#     for i in range(repeats):
#         shuff_df = shuffle(dataset)
#         target = shuff_df[target_column]
#         target_ls = kfold_split(target, num_folds)
#         fold_ls = kfold_split(shuff_df, num_folds)
#         # Model Training
#         for j in range(len(fold_ls)-1):
#             phifold = fold_ls[j]
#
#             model.fit()


import textwrap
def wrap_labels(ax, width, break_long_words=False):
    """
    https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce
    :param ax:
    :param width:
    :param break_long_words:
    :return:
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


