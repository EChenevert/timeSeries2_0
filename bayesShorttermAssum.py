import numpy as np
import pandas as pd
import main

data = main.load_data()
yearly = main.average_byyear_bysite_seasonal(data)
lasRates = yearly[yearly['Year (yyyy)'] > 2019].groupby('Simple site').median()  # [yearly['Year (yyyy)'] > 2019]

perc = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12006_PercentFlooded_CalendarYear\12006.csv",
                   encoding="unicode escape")
perc['Simple site'] = [i[:8] for i in perc['Station_ID']]
perc = perc.groupby('Simple site').median()
wl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12007_WaterLevelRange_CalendarYear\12007.csv",
                 encoding="unicode escape")
wl['Simple site'] = [i[:8] for i in wl['Station_ID']]
wl = wl.groupby('Simple site').median()

veg = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12008_VegPercentCover\12008.csv",
                  encoding="unicode escape").groupby('Site_ID').median()
marshElev = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12009_Survey_Marsh_Elevation\12009_Survey_Marsh_Elevation.csv",
                        encoding="unicode escape").groupby('SiteId').median()
geefrom2020 = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_GEE90percfrom2020.csv",
                          encoding="unicode escape")[['Simple_sit', 'NDVI', 'tss_med', 'windspeed']]\
    .groupby('Simple_sit').median()
distRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv",
                        encoding="unicode escape")[['Field1', 'distance_to_river_m']].groupby('Field1').median()
SEC = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12017_SurfaceElevation_ChangeRate\12017.csv",
                  encoding="unicode escape")
SEC['Simple site'] = [i[:8] for i in SEC['Station_ID']]
SEC = SEC.groupby('Simple site').median()
# Concatenate
df = pd.concat([lasRates, distRiver, geefrom2020, marshElev, veg, wl, perc, SEC], axis=1, join='inner')

# Make the subsidence and rslr variables: using the
df['Shallow Subsidence Rate (mm/yr)'] = df['Accretion Rate (mm/yr)'] - df['Surface Elevation Change Rate (cm/y)']*10
df['SEC Rate (mm/yr)'] = df['Surface Elevation Change Rate (cm/y)']*10
df['SLR (mm/yr)'] = 2.0  # from jankowski
df['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * df['Latitude']) - 114.26)*-1
df['RSLR (mm/yr)'] = df['Shallow Subsidence Rate (mm/yr)'] + df['Deep Subsidence Rate (mm/yr)'] + df['SLR (mm/yr)']

# Clean dataset
df = df.dropna(subset='Accretion Rate (mm/yr)')
df = df.dropna(thresh=df.shape[0]*0.9, how='all', axis=1)

dfi = df[[
    'RSLR (mm/yr)', 'Accretion Rate (mm/yr)', 'avg_flooding (ft)', '90%thUpper_flooding (ft)',
    '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)', 'avg_percentflooded (%)', 'distance_to_river_m',
    'NDVI', 'windspeed', 'Tide_Amp (ft)'
]]


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

outcome = 'Accretion Rate (mm/yr)'

# df = df.drop([
#     'width_mean', 'width_sd_m', 'width_med_', 'width_max_', 'width_min_'
# ], axis=1)

# Normalize dataset between 0 and 1
x_scaler = MinMaxScaler()
phi = x_scaler.fit_transform(df.drop(outcome, axis=1))

# Shuffle the dataset to avoid training in only certain spatial areas
np.random.shuffle(phi)

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(pd.DataFrame(phi, columns=df.drop(outcome, axis=1).columns.values))
plt.show()
# Seems to be kiiiinnnddaaa normal, barely w some variables

t = np.asarray(df[outcome])
# B, a, eff_lambda, itr = bml.iterative_prog_wPrior(phi, t, np.random.normal(5, 1, size=(len(phi[0, :]), 1)))  # std of 0.5 cuz i normalize variables between 0 and 1

# Find the weights with the obtainesd hyperparameters --> thru the lambda function
from sklearn.model_selection import train_test_split

MSE_map_ls = []
MSE_ml_ls = []
trainSize = []
trainFracArr = np.linspace(0.1, 0.9, 40)
for frac in trainFracArr:
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(phi[:, 1:], t, train_size=frac)  # 0 cuz 0 corresponds to the RSLR var
    print(len(X_train))
    B, a, eff_lambda, itr = bml.iterative_prog_wPrior(X_train, y_train,
                                                      phi[:, 0])   # 0 cuz 0 corresponds to the RSLR var
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

# NEXT: To either extract the predicted values (should be self explanatory from the returnMSE function) or to
# collect better data in order to put in this. An iterative CV evaluation should also be created







