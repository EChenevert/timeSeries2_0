import pandas as pd
import shap
from sklearn.model_selection import RepeatedKFold, cross_val_score, RandomizedSearchCV
import xgboost as xgb
import main
import funcs
import numpy as np

# This is to get the time series
dataframes = main.load_data()
avgSeasons = main.average_byyear_bysite_seasonal(dataframes)
# This is to get the time average / total site names
dataframes = main.load_data()
avgBysite = main.average_bysite(dataframes)
avgBysite = avgBysite.dropna(subset='Longitude').reset_index()
avgBysite[['Simple site', 'Latitude', 'Longitude', 'Basins', 'Community']]\
    .to_csv("D:\\Etienne\\fall2022\CRMS_data\\siteNamesAndCoords10_9.csv")

springWinterTS = avgSeasons[avgSeasons['Season'] == 2]
summerTS = avgSeasons[avgSeasons['Season'] == 1]

# Reduce into time averaged spring and winter sections
winterDF = springWinterTS.groupby('Simple site').median()
summerDF = summerTS.groupby('Simple site').median()

# Attach external Variables
# NDVI
summerNDVI = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\seasonalGEEdata\NDVIsummer_90perc.csv",
                         encoding='unicode_escape')[['Simple_sit', 'median']].set_index('Simple_sit')
summerNDVI = summerNDVI.rename(columns={'median': 'NDVI'})
winterNDVI = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\seasonalGEEdata\NDVIwinter_90perc.csv",
                         encoding='unicode_escape')[['Simple_sit', 'median']].set_index('Simple_sit')
winterNDVI = winterNDVI.rename(columns={'median': 'NDVI'})
# TSS
summerTSS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\seasonalGEEdata\TSSsummer_60perc.csv",
                        encoding='unicode_escape')[['Simple_sit', 'median']].set_index('Simple_sit')
summerTSS = summerTSS.rename(columns={'median': 'TSS'})
winterTSS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\seasonalGEEdata\TSSwinter_60perc.csv",
                        encoding='unicode_escape')[['Simple_sit', 'median']].set_index('Simple_sit')
winterTSS = winterTSS.rename(columns={'median': 'TSS'})
# GRWL data as well as distance to fluvial vectors
plusRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv", encoding='unicode_escape')[
    ['Field1', 'distance_to_river_m', 'width_mean', 'width_sd_m', 'width_med_', 'width_max_', 'width_min_']
].set_index('Field1')
# Distance to ocean boundary (sketchy)
distOcean = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\allDistOcean.csv", encoding='unicode_escape')[
    ['Field1', 'Distance_to_Ocean_m']
].set_index('Field1')

# Add hydrologic variables



# Concatenate to each
winterDF = pd.concat([winterDF, winterNDVI, winterTSS, plusRiver, distOcean], axis=1)
summerDF = pd.concat([summerDF, summerNDVI, summerTSS, plusRiver, distOcean], axis=1)

# Clean each of the datasets
winterDF = winterDF.dropna(subset='Accretion Rate (mm/yr)')
summerDF = summerDF.dropna(subset='Accretion Rate (mm/yr)')
winterDF = winterDF.dropna(thresh=winterDF.shape[0]*0.8, how='all', axis=1)
summerDF = summerDF.dropna(thresh=summerDF.shape[0]*0.8, how='all', axis=1)
# Drop unnessary columns
winterDF = winterDF.drop([
    'Average Accretion (mm)',  # may wanna check this oucome outcome
    'Year (yyyy)', 'Season', 'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)',
    'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)', 'Latitude', 'Longitude', 'Month (mm)',
    'Delta time (days)', 'Delta Time (decimal_years)', 'Measurement Depth (ft)',
    'Soil Porewater Temperature (°C)', 'Direction (Collar Number)', 'Direction (Compass Degrees)',
    'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)'
], axis=1)
summerDF = summerDF.drop([
    'Average Accretion (mm)',  # may wanna check this oucome outcome
    'Year (yyyy)', 'Season', 'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)',
    'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)', 'Latitude', 'Longitude', 'Month (mm)',
    'Delta time (days)', 'Delta Time (decimal_years)', 'Soil Moisture Content (%)', 'Bulk Density (g/cm3)',
    'Organic Matter (%)', 'Wet Volume (cm3)', 'Organic Density (g/cm3)', 'Measurement Depth (ft)',
    'Soil Porewater Temperature (°C)', 'Direction (Collar Number)', 'Direction (Compass Degrees)',
    'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)'
], axis=1)

dic = {'summer': summerDF, 'winter': winterDF}
for key in dic:
    print(key)
    dfout = funcs.outlierrm(dic[key])
    y = dfout['Accretion Rate (mm/yr)']
    X = dfout.drop(['Accretion Rate (mm/yr)'], axis=1)
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
    rs_model = RandomizedSearchCV(xgbmodel, param_distributions=params, n_iter=100, scoring='r2', n_jobs=-1, cv=5,
                                  verbose=1)
    rs_model.fit(X, y)
    bestxgb = rs_model.best_estimator_

    # Now use the selected features to create a model from the train data to test on the test data with repeated cv
    rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
    scores = cross_val_score(bestxgb, X, y.values.ravel(), scoring='r2',
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
        inds = shap.utils.potential_interactions(shap_values[:, feat],
                                                 shap_values)  # Explore what may interact with var idx1
        for i in range(3):  # take the 3 most important and plot
            shap.plots.scatter(shap_values[:, feat], color=shap_values[:, inds[i]])



