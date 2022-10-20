import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import funcs
import main
import numpy as np
import shap


# # TSS, NDVI
# crms60gee = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\CRMS_GEE60perc.csv", encoding='unicode_escape')[
#     ['Simple_sit', 'tss_med', 'NDVI', 'windspeed']
# ].set_index('Simple_sit')
# # GRWL data as well as distance to fluvial vectors
# plusRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv", encoding='unicode_escape')[
#     ['Field1', 'distance_to_river_m', 'width_mean', 'width_sd_m', 'width_med_', 'width_max_', 'width_min_']
# ].set_index('Field1')
# # Distance to ocean boundary (sketchy)
# distOcean = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\allDistOcean.csv", encoding='unicode_escape')[
#     ['Field1', 'Distance_to_Ocean_m']
# ].set_index('Field1')
# dfs = main.load_data()
# bysite = main.average_bysite(dfs)
#
# # Concatenate
# df = pd.concat([crms60gee, plusRiver, bysite, distOcean], axis=1, join='inner')
# df = df.dropna(subset='Accretion Rate (mm/yr)')
# df = df.dropna(thresh=df.shape[0]*0.9, how='all', axis=1)
# df = df.drop([
#     'Delta time (days)', 'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)', 'Year (yyyy)', 'Month (mm)',
#     'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)', 'Delta time (days)', 'Delta Time (decimal_years)',
#     'Wet Soil pH (pH units)', 'Dry Soil pH (pH units)', 'Wet Volume (cm3)', 'Dry Volume (cm3)', 'Organic Matter (%)',
#     'Organic Density (g/cm3)', 'Measurement Depth (ft)', 'Soil Porewater Temperature (°C)', 'Plot Size (m2)',
#     'Direction (Collar Number)', 'Direction (Compass Degrees)', 'Pin Number', 'Observed Pin Height (mm)',
#     'Verified Pin Height (mm)', 'Latitude', 'Longitude', 'Soil Moisture Content (%)',
#     'Bulk Density (g/cm3)', 'Average Accretion (mm)',  # 'Staff Gauge (ft)',
# ], axis=1)
#
#
# dfout = funcs.outlierrm(df)


# Using this dataset for direct comparison to bayesAnalsysis
dfout = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_dfi.csv", encoding="unicode escape")\
    .set_index('Unnamed: 0')

y = dfout['Accretion Rate (mm/yr)']
X = dfout.drop(['Accretion Rate (mm/yr)', 'RSLR (mm/yr)'], axis=1)
#
# x_scaler = MinMaxScaler()
# Xscaled = pd.DataFrame(x_scaler.fit_transform(X), columns=X.columns.values)

xgbmodel = xgb.XGBRegressor()
params = {
    'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    'min_child_weight': [1, 3, 5, 7, 9, 11],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
    'max_depth': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
    'n_estimators': [25, 50, 100, 400, 500, 600, 700, 800, 900, 1000],
    'reg_lambda': [0, 0.2, 0.4, 0.6, 0.8, 1],
    # 'sunsample': [0.2, 0.4, 0.6, 0.8, 1],
    'gamma': [0, 0.2, 0.4, 0.6, 0.8, 1]
}
# 624,000 grid space
rs_model = RandomizedSearchCV(xgbmodel, param_distributions=params, n_iter=100, scoring='neg_mean_squared_error', n_jobs=-1, cv=5,
                              verbose=1)
rs_model.fit(X, y)
bestxgb = rs_model.best_estimator_

# Now use the selected features to create a model from the train data to test on the test data with repeated cv
rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
scores = cross_val_score(bestxgb, X, y.values.ravel(), scoring='neg_mean_squared_error',
                         cv=rcv, n_jobs=-1)
rcvresults = scores
print('### BEST XBG WHOLE DATASET ###')
print(" mean RCV, and median RCV r2: ", np.mean(scores), np.median(scores))

# SHAP analysis
# add SHAPLEY
bestmodel = bestxgb
data = X
shap_values = shap.TreeExplainer(bestmodel).shap_values(data)
explainer = shap.Explainer(bestmodel)
shap_values = explainer(X)
# summarize the effects of all the features
shap.summary_plot(shap_values, features=data, feature_names=data.columns)
shap.summary_plot(shap_values, features=data, feature_names=data.columns, plot_type='bar')
# create a dependence scatter plot to show the effect of a single feature across the whole dataset
# kbestfeats.remove(outcome)
for feat in data.columns.values:  # 1: excludes the outcome:accretion
    inds = shap.utils.potential_interactions(shap_values[:, feat], shap_values)  # Explore what may interact with var idx1
    for i in range(3):  # take the 3 most important and plot
        shap.plots.scatter(shap_values[:, feat], color=shap_values[:, inds[i]])




