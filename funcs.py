import pandas as pd
from scipy import stats
import numpy as np
from sklearn.utils import shuffle
import random
import statsmodels.api as sm
from sklearn import linear_model
import textwrap


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


# https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/#:~:text=1.-,Forward%20selection,with%20all%20other%20remaining%20features.
def backward_elimination(data, target, num_feats=5, significance_level=0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level) or (len(features) > num_feats):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def unscaled_weights_from_full_standardization(X, y, bayesianReg: linear_model):
    """
    https://stackoverflow.com/questions/57513372/can-i-inverse-transform-the-intercept-and-coefficients-of-
    lasso-regression-after
    Better source:
    https://stats.stackexchange.com/questions/74622/converting-standardized-betas-back-to-original-variables
    """
    a = bayesianReg.coef_

    # Me tryna do my own thing
    coefs_new = []
    for x in range(len(X.columns)):
        # print(X.columns.values[x])
        col = X.columns.values[x]
        coefs_new.append(((a[x] * float(y.std())) / (np.asarray(X.std()[col]))))
    intercept = float(y.std()) - np.sum(np.multiply(np.asarray(coefs_new), np.asarray(X.mean())))  # hadamard product
    # print(intercept)
    # # First deal with X
    # mean_X = X.mean()  # is a pandas series of means
    # std_X = X.std()
    # # X_scaled = ((X - mean_X) / std_X)
    # # Second deal with y: for computing the intercept
    # mean_y = y.median()
    # std_y = y.std()
    # coef_new = ((a * (X - mean_X).values) / (X * std_X).values) * float(std_y)
    # coef_new = np.nan_to_num(coef_new)[0]  # need this cuz some values of X will be zero; [0] just takes first arr
    # b = bayesianReg.intercept_
    # # The intercept is expected accretion when all Xs are 0: so ideal this is zero or close to zero.
    # # If not we mising some process
    # intercept_new = b * float(std_y) + float(mean_y)

    return coefs_new, intercept

def log_transform_weights(coefs):
    exp_coefs = np.exp(coefs)
    # print(" exponential ", exp_coefs)
    minus_coefs = exp_coefs - 1
    # minus_coefs = [i - 1 for i in exp_coefs]
    # print("Minus 1: ", minus_coefs)
    new_coefs = minus_coefs * 100
    # print(" multiply 100: ", new_coefs)
    return new_coefs

# log_transform_weights(unscaled_weights)

