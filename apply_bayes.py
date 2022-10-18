import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import bayes_ml_funcs as bml


# 1.) Aquire dataset I want to work with
# ------------> may have to be done after exploratory analysis / feature selection process
# 2.) Normalize dataset between -1 and 1
# 3.) Use iterative program to find hyperparameters
# ------------> remember that this is done by

test = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\testDataset.csv", encoding='unicode_escape')\
    .set_index('Unnamed: 0').dropna()
outcome = 'Accretion Rate (mm/yr)'

test = test.drop([
    'width_mean', 'width_sd_m', 'width_med_', 'width_max_', 'width_min_'
], axis=1)
# Normalize dataset between 0 and 1
x_scaler = MinMaxScaler()
phi = x_scaler.fit_transform(test.drop(outcome, axis=1))

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(pd.DataFrame(phi, columns=test.drop(outcome, axis=1).columns.values))
plt.show()
# Seems to be kiiiinnnddaaa normal, barely w some variables

t = np.asarray(test[outcome])
B, a, eff_lambda, itr = bml.iterative_prog_wPrior(phi, t, np.random.normal(5, 1, size=(len(phi[0, :]), 1)))  # std of 0.5 cuz i normalize variables between 0 and 1

# Find the weights with the obtainesd hyperparameters --> thru the lambda function
var_weights_map_ls = []
var_weights_ml_ls = []
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    var_weights_map = bml.regLn(eff_lambda, phi, t)
    map_MSE = bml.returnMSE(phi, var_weights_map, t)
    var_weights_map_ls.append(map_MSE)
    var_weights_ml = bml.regLn(0, phi, t)  # recall that ml is when lambda is 0
    ml_MSE =bml.returnMSE(phi, var_weights_ml, t)
    var_weights_ml_ls.append(ml_MSE)





