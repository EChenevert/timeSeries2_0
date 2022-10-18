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

# Shuffle the dataset to avoid training in only certain spatial areas
np.random.shuffle(phi)

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(pd.DataFrame(phi, columns=test.drop(outcome, axis=1).columns.values))
plt.show()
# Seems to be kiiiinnnddaaa normal, barely w some variables

t = np.asarray(test[outcome])
# B, a, eff_lambda, itr = bml.iterative_prog_wPrior(phi, t, np.random.normal(5, 1, size=(len(phi[0, :]), 1)))  # std of 0.5 cuz i normalize variables between 0 and 1

# Find the weights with the obtainesd hyperparameters --> thru the lambda function
from sklearn.model_selection import train_test_split

MSE_map_ls = []
MSE_ml_ls = []
trainSize = []
trainFracArr = np.linspace(0.1, 0.9, 40)
for frac in trainFracArr:
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(phi, t, train_size=frac)
    print(len(X_train))
    B, a, eff_lambda, itr = bml.iterative_prog_wPrior(X_train, y_train,
                                                      np.random.normal(12, 1, size=(len(X_train[0, :]), 1)))  # std of 0.5 cuz i normalize variables between 0 and 1
    # B, a, eff_lambda, itr = bml.iterative_prog(X_train, y_train)  # std of 0.5 cuz i normalize variables between 0 and 1

    var_weights_map = bml.regLn(eff_lambda, X_train, y_train)
    map_MSE = bml.returnMSE(X_test, var_weights_map, y_test)
    MSE_map_ls.append(map_MSE)
    var_weights_ml = bml.regLn(0, X_train, y_train)  # recall that ml is when lambda is 0
    ml_MSE = bml.returnMSE(X_train, var_weights_ml, y_train)
    MSE_ml_ls.append(ml_MSE)
    # Append train size for plotting
    trainSize.append(frac)

plt.figure()
plt.plot(trainSize, MSE_map_ls, label='map MSE')
plt.plot(trainSize, MSE_ml_ls, label='ml MSE')
plt.title('MSE versus Train Size')
plt.ylabel('MSE')
plt.xlabel('Train Size')
plt.legend()
plt.show()
# To me this plot seems to say that test-train splits should not be used for model evaluation because I
# seem to have a lot of variability in the MSE

# NEXT: To either extract the predicted values (should be self explanatory from the returnMSE function)

