import pandas as pd
from scipy import stats
import numpy as np


def outlierrm(df, thres=3):
    """Dont put in long lats in here! Need Year and Site name lol"""
    df = df.dropna()  #.set_index(['Simple site', 'level_1'])
    switch = False
    if 'Basins' in df.columns.values or 'Community' in df.columns.values:
        print('True')
        switch = True
        holdstrings = df[['Basins', 'Community']]
        df = df.drop(['Basins', 'Community'], axis=1)
    df = df.apply(pd.to_numeric)
    length = len(df.columns.values)
    for col in df.columns.values:
        df[col + "_z"] = stats.zscore(df[col])
    for col in df.columns.values[length-1:]:
        df = df[np.abs(df[col]) < thres]
    df = df.drop(df.columns.values[length:], axis=1)
    if switch:
        df = pd.concat([df, holdstrings], join='inner', axis=1)
    return df