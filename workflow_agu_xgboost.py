from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model

import main
import pandas as pd
import numpy as np


# Everything I need for this should be within the file "D:\Etienne\fall2022\agu_data"
## Data from CIMS
data = main.load_data()
yearly = main.average_byyear_bysite_seasonal(data)
bysite = yearly.groupby('Simple site').median()

## Data from CRMS
perc = pd.read_csv(r"D:\Etienne\fall2022\agu_data\percentflooded.csv",
                   encoding="unicode escape")
perc['Simple site'] = [i[:8] for i in perc['Station_ID']]
perc = perc.groupby('Simple site').median()
wl = pd.read_csv(r"D:\Etienne\fall2022\agu_data\waterlevelrange.csv",
                 encoding="unicode escape")
wl['Simple site'] = [i[:8] for i in wl['Station_ID']]
wl = wl.groupby('Simple site').median()

marshElev = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12009_Survey_Marsh_Elevation\12009_Survey_Marsh_Elevation.csv",
                        encoding="unicode escape").groupby('SiteId').median().drop('Unnamed: 4', axis=1)
SEC = pd.read_csv(r"D:\Etienne\fall2022\agu_data\12017_SurfaceElevation_ChangeRate\12017.csv",
                  encoding="unicode escape")
SEC['Simple site'] = [i[:8] for i in SEC['Station_ID']]
SEC = SEC.groupby('Simple site').median().drop('Unnamed: 4', axis=1)

acc = pd.read_csv(r"D:\Etienne\fall2022\agu_data\12172_SEA\Accretion__rate.csv", encoding="unicode_escape")[
    ['Site_ID', 'Acc_rate_fullterm (cm/y)']
].groupby('Site_ID').median()

## Data from Gee and Arc
jrc = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_GEE_JRCCOPY2.csv", encoding="unicode_escape")[
    ['Simple_sit', 'Land_Lost_m2']
].set_index('Simple_sit')

gee = pd.read_csv(r"D:\Etienne\fall2022\agu_data\CRMS_GEE60pfrom2007to2022.csv",
                          encoding="unicode escape")[['Simple_sit', 'NDVI', 'tss_med', 'windspeed']]\
    .groupby('Simple_sit').median().fillna(0)  # filling nans with zeros cuz all nans are in tss because some sites are not near water
distRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv",
                        encoding="unicode escape")[['Field1', 'distance_to_river_m', 'width_mean']].groupby('Field1').median()
nearWater = pd.read_csv(r"D:\Etienne\fall2022\agu_data\ALLDATA2.csv", encoding="unicode_escape")[
    ['Simple site', 'Distance_to_Water_m']
].set_index('Simple site')


# Concatenate
df = pd.concat([bysite, distRiver, nearWater, gee, jrc, marshElev, wl, perc, SEC, acc], axis=1, join='outer')

# Now clean the columns
# First delete columns that are more than 1/2 nans
tdf = df.dropna(thresh=df.shape[0]*0.5, how='all', axis=1)
# Drop uninformative features
udf = tdf.drop([
    'Year (yyyy)', 'Season', 'Accretion Measurement 1 (mm)', 'Year',
    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    'Accretion Measurement 4 (mm)', 'Longitude',
    'Month (mm)', 'Average Accretion (mm)', 'Delta time (days)', 'Wet Volume (cm3)',
    'Delta Time (decimal_years)', 'Wet Soil pH (pH units)', 'Dry Soil pH (pH units)', 'Dry Volume (cm3)',
    'Measurement Depth (ft)', 'Plot Size (m2)', '% Cover Shrub', '% Cover Carpet', 'Direction (Collar Number)',
    'Direction (Compass Degrees)', 'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
    'calendar_year', 'percent_waterlevel_complete',
    'Average Height Shrub (cm)', 'Average Height Carpet (cm)'  # I remove these because most values are nan and these vars are unimportant really
], axis=1)



# Address the vertical measurement for mass calculation (multiple potential outcome problem)
vertical = 'Accretion Rate (mm/yr)'
if vertical == 'Accretion Rate (mm/yr)':
    udf = udf.drop('Acc_rate_fullterm (cm/y)', axis=1)
    # Make sure multiplier of mass acc is in the right units
    udf['Average_Ac_cm_yr'] = udf['Accretion Rate (mm/yr)'] / 10  # mm to cm conversion
    # Make sure subsidence and RSLR are in correct units
    udf['Shallow Subsidence Rate (mm/yr)'] = udf[vertical] - udf['Surface Elevation Change Rate (cm/y)'] * 10
    udf['Shallow Subsidence Rate (mm/yr)'] = [0 if val < 0 else val for val in udf['Shallow Subsidence Rate (mm/yr)']]
    udf['SEC Rate (mm/yr)'] = udf['Surface Elevation Change Rate (cm/y)'] * 10
    # Now calcualte subsidence and RSLR
    # Make the subsidence and rslr variables: using the
    udf['SLR (mm/yr)'] = 2.0  # from jankowski
    udf['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * udf['Latitude']) - 114.26) * -1
    udf['RSLR (mm/yr)'] = udf['Shallow Subsidence Rate (mm/yr)'] + udf['Deep Subsidence Rate (mm/yr)'] + udf[
        'SLR (mm/yr)']
    udf = udf.drop(['SLR (mm/yr)', 'Latitude'],
                   axis=1)  # obviously drop because it is the same everywhere ; only used for calc

elif vertical == 'Acc_rate_fullterm (cm/y)':
    udf = udf.drop('Accretion Rate (mm/yr)', axis=1)
    #  Make sure multiplier of mass acc is in the right units
    udf['Average_Ac_cm_yr'] = udf[vertical]
    # Make sure subsidence and RSLR are in correct units
    udf['Shallow Subsidence Rate (mm/yr)'] = (udf[vertical] - udf['Surface Elevation Change Rate (cm/y)'])*10
    udf['SEC Rate (cm/yr)'] = udf['Surface Elevation Change Rate (cm/y)']
    # Now calcualte subsidence and RSLR
    # Make the subsidence and rslr variables: using the
    udf['SLR (mm/yr)'] = 2.0  # from jankowski
    udf['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * udf['Latitude']) - 114.26) * -1
    udf['RSLR (mm/yr)'] = udf['Shallow Subsidence Rate (mm/yr)'] + udf['Deep Subsidence Rate (mm/yr)'] + udf[
        'SLR (mm/yr)']*0.1
    udf = udf.drop(['SLR (mm/yr)', 'Latitude'],
                   axis=1)  # obviously drop because it is the same everywhere ; only used for calc
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


A = 10000  # This is the area of the study, in our case it is per site, so lets say the area is 1 m2 in cm
udf['Total Mass Accumulation (g/yr)'] = (udf['Bulk Density (g/cm3)'] * udf['Average_Ac_cm_yr']) * A  # g/cm3 * cm/yr * cm2 = g/yr
udf['Organic Mass Accumulation (g/yr)'] = (udf['Bulk Density (g/cm3)'] * udf['Average_Ac_cm_yr'] * (udf['Organic Matter (%)']/100)) * A
udf['Mineral Mass Accumulation (g/yr)'] = udf['Total Mass Accumulation (g/yr)'] - udf['Organic Mass Accumulation (g/yr)']
# Just drop the terms to be safew
udf = udf.drop([vertical, 'Average_Ac_cm_yr', 'Total Mass Accumulation (g/yr)'], axis=1)

# ########### Define outcome ########## SWITCH BETWEEN ORGANIC MASS ACC AND MINERLA MASS ACC
outcome = 'Organic Mass Accumulation (g/yr)'
if outcome == 'Mineral Mass Accumulation (g/yr)':
    udf = udf.drop('Organic Mass Accumulation (g/yr)', axis=1)
elif outcome == 'Organic Mass Accumulation (g/yr)':
    udf = udf.drop('Mineral Mass Accumulation (g/yr)', axis=1)
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', 'Soil Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'tss_med_mg/l'})

# conduct outlier removal which drops all nans
import funcs
rdf = funcs.outlierrm(udf, thres=3)

# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
rdf['log_river_width_mean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
rdf['log_distance_to_river_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km', 'river_width_mean_km'], axis=1)

# Now it is feature selection time
# drop any variables related to the outcome
rdf = rdf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    'Soil Moisture Content (%)', 'Bulk Density (g/cm3)', 'Organic Matter (%)', 'Organic Density (g/cm3)',
    'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
    # taking out water level features because they are not super informative
    '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
    'Staff Gauge (ft)',
    'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    'log_river_width_mean_km'  # i just dont like this variable because it has a sucky distribution
], axis=1)


# Time to build the XGBoost model! no feature selection needed!
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, RandomizedSearchCV
import xgboost as xgb

target = rdf[outcome].reset_index().drop('index', axis=1)
predictors = rdf.drop([outcome], axis=1).reset_index().drop('index', axis=1)

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.5, shuffle=True, random_state=1)

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
rs_model.fit(X_train, y_train)  # even tho for cross_val_score no splitting needs to be done, I need to plit to make unbiased model wrt to hyperparams
bestxgb = rs_model.best_estimator_

# Now use the selected features to create a model from the train data to test on the test data with repeated cv
# REMEBER cross val score trains the model a new each time
rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
scores = cross_val_score(bestxgb, X_test, y_test.values.ravel(), scoring='r2',
                         cv=rcv, n_jobs=-1)
rcvresults = scores
print('### BEST XBG WHOLE DATASET ###')
print(" mean RCV, and median RCV r2: ", np.mean(scores), np.median(scores))

# SHAP analysis
import shap
# add SHAPLEY
bestmodel = bestxgb
data = X_test
shap_values = shap.TreeExplainer(bestmodel).shap_values(data)
explainer = shap.Explainer(bestmodel)
shap_values = explainer(data)
# summarize the effects of all the features
shap.summary_plot(shap_values, features=data, feature_names=data.columns)


