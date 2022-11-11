from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model

import main
import pandas as pd
import numpy as np


# Everything I need for this should be within the file "D:\Etienne\fall2022\agu_data"
## Data from CIMS
data = main.load_data()
bysite = main.average_bysite(data)


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
    'Year (yyyy)', 'Accretion Measurement 1 (mm)', 'Year',
    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    'Accretion Measurement 4 (mm)', 'Longitude', 'Basins',
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
    # udf['Average_Ac_cm_yr'] = udf['Accretion Rate (mm/yr)'] / 10  # mm to cm conversion
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
    # udf['Average_Ac_cm_yr'] = udf[vertical]
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

####### Define outcome as vertical component
outcome = vertical

# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', 'Soil Specific Conductance (uS/cm)',
                'Soil Porewater Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'tss_med_mg/l'})

# conduct outlier removal which drops all nans
import funcs
rdf = funcs.outlierrm(udf.drop('Community', axis=1), thres=3)

# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
rdf['log_river_width_mean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
rdf['log_distance_to_river_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km', 'river_width_mean_km'], axis=1)
# Now it is feature selection time
# drop any variables related to the outcome
rdf = rdf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
    # taking out water level features because they are not super informative
    '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
    'Staff Gauge (ft)',
    'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    'log_river_width_mean_km'  # i just dont like this variable because it has a sucky distribution
], axis=1)


# Now for actual feature selection yay!!!!!!!!!!!!!!!!!!!!!!!!!!
# Make Dataset
target = rdf[outcome].reset_index().drop('index', axis=1)
predictors = rdf.drop([outcome], axis=1).reset_index().drop('index', axis=1)
# NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
mlr = linear_model.LinearRegression()
# l = linear_model.Lasso()
feature_selector = ExhaustiveFeatureSelector(mlr,
                                             min_features=1,
                                             max_features=5,  # I should only use 5 features (15 takes waaaaay too long)
                                             scoring='r2',  # minimizes variance, at expense of bias
                                             # print_progress=True,
                                             cv=5)  # 5 fold cross-validation

efsmlr = feature_selector.fit(predictors, target.values.ravel())  # these are not scaled... to reduce data leakage

print('Best CV r2 score: %.2f' % efsmlr.best_score_)
print('Best subset (indices):', efsmlr.best_idx_)
print('Best subset (corresponding names):', efsmlr.best_feature_names_)

bestfeatures = list(efsmlr.best_feature_names_)

# Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val
#### NOTE: I should do separate tests to determine which split of the data is optimal ######
# first split data set into test train
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

X, y = predictors[bestfeatures], target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)


br = linear_model.BayesianRidge(fit_intercept=False, tol=10e-5)
br.fit(X, y)
# Lets check those hyperparameters: lambda corresponds to precision over parameters... alpha is precision over posterior
print("lambda (I know as alpha): ", br.lambda_)
print("alpha (I know as beta): ", br.alpha_)
# lets check the estimates of the w weight vector (from ML derived from least squares solution)
print("learned weight vector: ", br.coef_)
print(X.columns.values)
# Lets check out the training model score
trainscore = br.score(X, y)
print("Training Score is: ", trainscore)
# Predictions
# So...... the predictions with this are weird.... we can only get a score that corresponds to R^2 it seems
# But that seems weird because I have the weights.... can't I just compute the point estimates?
#
ypred, stdpred = br.predict(X, return_std=True)  # the y_pred is the mean of the pred_dist for that sample, the stdpred is the std for that sample

from sklearn.metrics import r2_score, mean_absolute_error
mae = mean_absolute_error(y, ypred)
r2 = r2_score(y, ypred)
print("Test MAE: ", mae)
print("Test R^2: ", r2)

# Do cross validation on whole dataset: cross val score fits the data each time to the inputted model, leaving some out and testing it against that left out
# the splitting above was only for a test train split test (just for fun but below is more accurate)
rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
scores = cross_val_score(br, X, y.values.ravel(), cv=rcv, scoring='r2')
print("Mean & median r2 repeated cross val: ", np.mean(scores), "  ", np.median(scores))

# So now we have to use shap to make sure that we interpret the model correctly (due to scaling probs and see the mean centered influences)
# the coeffiencets themselves are zeros centered

# SHAP analysis
import shap
# add SHAPLEY
data = X  # decided to use X_test because I wanted it to be on NEW data that the model was not fit too;

masker = shap.maskers.Independent(data=data)

explainer = shap.Explainer(
    br, masker=masker, feature_names=data.columns
)
sv = explainer(data)
shap.summary_plot(sv, features=data, feature_names=data.columns, plot_type='bar')

# Do dependence plots for these guys
for var in data.columns.values:
    # Dependence plots
    shap.partial_dependence_plot(
        var, br.predict, data, ice=False,
        model_expected_value=True, feature_expected_value=True
    )
    # correposnding shap plots
    shap.plots.scatter(sv[:, var])



# so it doesn't really work on the whole dataset
# lets break into groups

gdf = pd.concat([rdf, udf['Community']], axis=1, join='inner')

# split into marsh datasets

brackdf = gdf[gdf['Community'] == 'Brackish']
saldf = gdf[gdf['Community'] == 'Saline']
freshdf = gdf[gdf['Community'] == 'Freshwater']
interdf = gdf[gdf['Community'] == 'Intermediate']
# Exclude swamp
marshdic = {'Brackish': brackdf, 'Saline': saldf, 'Freshwater': freshdf, 'Intermediate': interdf}

preddic = {}
for key in marshdic:
    print(key)
    mdf = marshdic[key]  # .drop('Community', axis=1)
    # It is preshuffled so i do not think ordering will be a problem
    target = mdf[outcome].reset_index().drop('index', axis=1)
    predictors = mdf.drop([outcome, 'Community'], axis=1).reset_index().drop('index', axis=1)
    # NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
    mlr = linear_model.LinearRegression()
    # l = linear_model.Lasso()
    feature_selector = ExhaustiveFeatureSelector(mlr,
                                                 min_features=1,
                                                 max_features=5,
                                                 # I should only use 5 features (15 takes waaaaay too long)
                                                 scoring='r2',  # minimizes variance, at expense of bias
                                                 # print_progress=True,
                                                 cv=5)  # 5 fold cross-validation

    efsmlr = feature_selector.fit(predictors, target.values.ravel())  # these are not scaled... to reduce data leakage

    print('Best CV r2 score: %.2f' % efsmlr.best_score_)
    print('Best subset (indices):', efsmlr.best_idx_)
    print('Best subset (corresponding names):', efsmlr.best_feature_names_)

    bestfeaturesM = list(efsmlr.best_feature_names_)

    # Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val
    #### NOTE: I should do separate tests to determine which split of the data is optimal ######
    # first split data set into test train
    from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

    X, y = predictors[bestfeaturesM], target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)

    br = linear_model.BayesianRidge(fit_intercept=False, tol=10e-5)
    br.fit(X, y)
    # Lets check those hyperparameters: lambda corresponds to precision over parameters... alpha is precision over posterior
    print("lambda (I know as alpha): ", br.lambda_)
    print("alpha (I know as beta): ", br.alpha_)
    # lets check the estimates of the w weight vector (from ML derived from least squares solution)
    print("learned weight vector: ", br.coef_)
    print(X.columns.values)
    # Lets check out the training model score
    trainscore = br.score(X, y)
    print("Training Score is: ", trainscore)
    # Predictions
    # So...... the predictions with this are weird.... we can only get a score that corresponds to R^2 it seems
    # But that seems weird because I have the weights.... can't I just compute the point estimates?
    #
    ypred, stdpred = br.predict(X,
                                return_std=True)  # the y_pred is the mean of the pred_dist for that sample, the stdpred is the std for that sample

    # save standard deviations
    preddic[key] = stdpred

    from sklearn.metrics import r2_score, mean_absolute_error

    mae = mean_absolute_error(y, ypred)
    r2 = r2_score(y, ypred)
    print("Test MAE: ", mae)
    print("Test R^2: ", r2)

    # Do cross validation on whole dataset: cross val score fits the data each time to the inputted model, leaving some out and testing it against that left out
    # the splitting above was only for a test train split test (just for fun but below is more accurate)
    rcv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1)
    scores = cross_val_score(br, X, y.values.ravel(), cv=rcv, scoring='r2')
    print("Mean & median r2 repeated cross val: ", np.mean(scores), "  ", np.median(scores))

    # So now we have to use shap to make sure that we interpret the model correctly (due to scaling probs and see the mean centered influences)
    # the coeffiencets themselves are zeros centered

    # # SHAP analysis
    # import shap
    #
    # # add SHAPLEY
    # data = X_test  # decided to use X_test because I wanted it to be on NEW data that the model was not fit too;
    #
    # masker = shap.maskers.Independent(data=data)
    #
    # explainer = shap.Explainer(
    #     br, masker=masker, feature_names=data.columns
    # )
    # sv = explainer(data)
    # shap.summary_plot(sv, features=data, feature_names=data.columns, plot_type='bar')
    #
    # # Do dependence plots for these guys
    # for var in data.columns.values:
    #     # Dependence plots
    #     shap.partial_dependence_plot(
    #         var, br.predict, data, ice=False,
    #         model_expected_value=True, feature_expected_value=True
    #     )
    #     # correposnding shap plots
    #     shap.plots.scatter(sv[:, var])


# plot box plots of prediction std distributions
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.boxplot(preddic.values())
ax.set_xticklabels(preddic.keys())
plt.title('Bayesian Uncertainty Plot by Marsh type')
plt.ylabel('Variance of Prediction Distribution')
plt.xlabel("Marsh Type")
plt.show()
