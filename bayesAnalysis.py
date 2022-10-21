
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(dfi['Accretion Rate (mm/yr)'], dfi['SEC Rate (mm/yr)'])
# plt.show()

#### NEXT IS TO CREATE THAT BAYSIAN LINEAR REGRESSION FROM THE BAYES FUNCS FILE!!!!!!!!!!!!!!


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import bayes_ml_funcs as bml


# 1.) Aquire dataset I want to work with
# ------------> may have to be done after exploratory analysis / feature selection process
# 2.) Normalize dataset between -1 and 1
# 3.) Use iterative program to find hyperparameters
# ------------> remember that this is done by

df = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_dfi.csv", encoding="unicode escape")\
    .set_index('Unnamed: 0')
outcome = 'Accretion Rate (mm/yr)'
df_vars = df.columns.values

df['distance_to_river_m'] = np.log(df['distance_to_river_m'])
# df['Tide_Amp (ft)'] = np.log(df['Tide_Amp (ft)'])
df = df.drop('Tide_Amp (ft)', axis=1)
# Normality test


# Normalize dataset between 0 and 1
x_scaler = MinMaxScaler((0, 1))
# x_scaler = MinMaxScaler((-1, 1))

phi = x_scaler.fit_transform(df.drop(outcome, axis=1))

# Shuffle the dataset to avoid training in only certain spatial areas
np.random.shuffle(phi)

import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(pd.DataFrame(phi, columns=df.drop(outcome, axis=1).columns.values))
# plt.show()
# Seems to be kiiiinnnddaaa normal, barely w some variables

t = np.asarray(df[outcome])
# B, a, eff_lambda, itr = bml.iterative_prog_wPrior(phi, t, np.random.normal(5, 1, size=(len(phi[0, :]), 1)))  # std of 0.5 cuz i normalize variables between 0 and 1

# Find the weights with the obtainesd hyperparameters --> thru the lambda function
from sklearn.model_selection import train_test_split

MSE_map_ls = []
MSE_ml_ls = []
MSE_map_winfo = []
trainSize = []
trainFracArr = np.linspace(0.1, 0.9, 40)
for frac in trainFracArr:
    hold_mlMSE = []
    hold_mapMSE = []
    hold_mapMSE_winfo = []
    for i in range(100):
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(phi[:, 1:], t, train_size=frac)  # 0 cuz 0 corresponds to the RSLR var
        # B, a, eff_lambda_winfo, itr = bml.iterative_prog_wPrior(X_train, y_train,
        #                                                   phi[:, 0])   # 0 cuz 0 corresponds to the RSLR var
        B, a, eff_lambda, itr = bml.iterative_prog(X_train, y_train)  # std of 0.5 cuz i normalize variables between 0 and 1

        # var_weights_map_winfo = bml.leastSquares(eff_lambda_winfo, X_train, y_train)
        # map_MSE_winfo = bml.returnMSE(X_test, var_weights_map_winfo, y_test)
        # hold_mapMSE_winfo.append(map_MSE_winfo)

        var_weights_map = bml.leastSquares(eff_lambda, X_train, y_train)
        map_MSE = bml.returnMSE(X_test, var_weights_map, y_test)
        hold_mapMSE.append(map_MSE)

        var_weights_ml = bml.leastSquares(0, X_train, y_train)  # recall that ml is when lambda is 0
        ml_MSE = bml.returnMSE(X_test, var_weights_ml, y_test)
        hold_mlMSE .append(ml_MSE)
    # Append train size for plotting
    trainSize.append(frac)
    MSE_ml_ls.append(np.mean(hold_mlMSE))
    MSE_map_ls.append(np.mean(hold_mapMSE))
    # MSE_map_winfo.append(np.mean(hold_mapMSE_winfo))

plt.figure()
plt.plot(trainSize, MSE_map_ls, label='MAP')
plt.plot(trainSize, MSE_ml_ls, label='MLE')
# plt.plot(trainSize, MSE_map_winfo, label='informed MAP')
# plt.ylim(0, 500)
plt.title('MSE versus Train Size')
plt.ylabel('MSE')
plt.xlabel('Train Size')
# plt.ylim(0, 500)
plt.legend()
plt.show()
# To me this plot seems to say that test-train splits should not be used for model evaluation because I
# seem to have a lot of variability in the MSE

# NEXT: To either extract the predicted values (should be self explanatory from the returnMSE function) or to
# collect better data in order to put in this. An iterative CV evaluation should also be created







