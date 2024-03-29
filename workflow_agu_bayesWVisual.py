from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model
import matplotlib.pyplot as plt
import main
import pandas as pd
import numpy as np
import funcs
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, cross_val_predict, \
    cross_validate, KFold

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
    '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)', 'std_deviation_water_level(ft NAVD88)',
    'Staff Gauge (ft)',
    'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution
    'Bulk Density (g/cm3)',  'Organic Density (g/cm3)'
    # 'Soil Porewater Temperature (°C)', 'Soil Moisture Content (%)', 'Organic Matter (%)',
], axis=1)


# Now for actual feature selection yay!!!!!!!!!!!!!!!!!!!!!!!!!!
# Make Dataset
target = rdf[outcome].reset_index().drop('index', axis=1)
predictors = rdf.drop([outcome], axis=1).reset_index().drop('index', axis=1)
#### Scale: Because this way I can extract feature importances
from sklearn.preprocessing import MinMaxScaler
scalar_whole = MinMaxScaler()
predictors = pd.DataFrame(scalar_whole.fit_transform(predictors), columns=predictors.columns.values)

# NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
br = linear_model.BayesianRidge()
feature_selector = ExhaustiveFeatureSelector(br,
                                             min_features=1,
                                             max_features=5,  # I should only use 5 features (15 takes waaaaay too long)
                                             scoring='neg_root_mean_squared_error',  # minimizes variance, at expense of bias
                                             # print_progress=True,
                                             cv=3)  # 5 fold cross-validation

efsmlr = feature_selector.fit(predictors, target.values.ravel())  # these are not scaled... to reduce data leakage

print('Best CV r2 score: %.2f' % efsmlr.best_score_)
print('Best subset (indices):', efsmlr.best_idx_)
print('Best subset (corresponding names):', efsmlr.best_feature_names_)

bestfeatures = list(efsmlr.best_feature_names_)

# Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val
#### NOTE: I should do separate tests to determine which split of the data is optimal ######
# first split data set into test train

X, y = predictors[bestfeatures], target

baymod = linear_model.BayesianRidge()  #.LinearRegression()  #Lasso(alpha=0.1)

# Now use the selected features to create a model from the train data to test on the test data with repeated cv
# just checking if repeated is in line with simple 3 fold cross val
rcv = RepeatedKFold(n_splits=3, n_repeats=100, random_state=1)
scores_for_repeated = cross_validate(baymod, X, y.values.ravel(), cv=rcv, scoring=('r2', 'neg_mean_absolute_error'),
                                     n_jobs=-1)

print('#### Bayesian Regression MODEL: 100x 3-Fold split')
print(" mean RCV, and median RCV r2: ", np.mean(scores_for_repeated['test_r2']), np.median(scores_for_repeated['test_r2']))
print(" mean RCV, and median RCV mae: ", np.mean(scores_for_repeated['test_neg_mean_absolute_error']),
      np.median(scores_for_repeated['test_neg_mean_absolute_error']))

# This RCV picks the best model from the repeated 3fold CV
gridsearcher = GridSearchCV(baymod, param_grid={}, cv=rcv, scoring='neg_root_mean_squared_error')
gridsearcher.fit(X, y.values.ravel())
best_br = gridsearcher.best_estimator_
alldata_dic = {'weights': best_br.coef_, 'features': bestfeatures, 'alpha': best_br.alpha_, 'lambda': best_br.lambda_,
               'sigma': best_br.sigma_}
## Try to get the number of determined parameters here from the sigma ....


################ Scores for basic model generalization (using whole dataset for cross val) ###############################
cv = KFold(n_splits=3)
scores_for_pred = cross_validate(baymod, X, y.values.ravel(), cv=cv, scoring=('r2', 'neg_mean_absolute_error'), n_jobs=-1)

print('#### Bayesian Regression MODEL: 3 Fold split ')
print(" mean RCV, and median RCV r2: ", np.mean(scores_for_pred['test_r2']), np.median(scores_for_pred['test_r2']))
print(" mean RCV, and median RCV mae: ", np.mean(scores_for_pred['test_neg_mean_absolute_error']),
      np.median(scores_for_pred['test_neg_mean_absolute_error']))

# Visualize the data
predicted = cross_val_predict(baymod, X, y.values.ravel(), cv=cv)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.title("Whole Dataset: 3-fold")
plt.show()


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
hold_alphas = {}
marsh_params_dic = {}
for key in marshdic:
    print(key)
    mdf = marshdic[key]  # .drop('Community', axis=1)
    # It is preshuffled so i do not think ordering will be a problem
    target = mdf[outcome].reset_index().drop('index', axis=1)
    predictors = mdf.drop([outcome, 'Community'], axis=1).reset_index().drop('index', axis=1)
    # Scale: because I want feature importances
    scalar_marsh = MinMaxScaler()
    predictors = pd.DataFrame(scalar_marsh.fit_transform(predictors), columns=predictors.columns.values)
    # NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
    # mlr = linear_model.LinearRegression()
    br = linear_model.BayesianRidge()

    feature_selector = ExhaustiveFeatureSelector(br,
                                                 min_features=1,
                                                 max_features=5,
                                                 # I should only use 5 features (15 takes waaaaay too long)
                                                 scoring='neg_root_mean_squared_error',
                                                 # print_progress=True,
                                                 cv=3)  # 5 fold cross-validation

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

    baymod = linear_model.BayesianRidge()  #.LinearRegression()  #Lasso(alpha=0.1)

    # Now use the selected features to create a model from the train data to test on the test data with repeated cv
    rcv = RepeatedKFold(n_splits=3, n_repeats=100, random_state=1)
    scores_repeated_marsh = cross_validate(baymod, X, y.values.ravel(), cv=rcv, scoring=('r2', 'neg_mean_absolute_error'), n_jobs=-1)
    # scores = cross_validate(baymod, X, y.values.ravel(), cv=rcv, scoring=('r2', 'neg_mean_absolute_error'), n_jobs=-1)
    print('#### Bayesian Regression MODEL: Repeated 3-Fold results')
    print(" mean RCV, and median RCV r2: ", np.mean(scores_repeated_marsh['test_r2']),
          np.median(scores_repeated_marsh['test_r2']))
    print(" mean RCV, and median RCV mae: ", np.mean(scores_repeated_marsh['test_neg_mean_absolute_error']),
          np.median(scores_repeated_marsh['test_neg_mean_absolute_error']))

    # This RCV picks the best model from the repeated 3fold CV
    gridsearcher = GridSearchCV(baymod, param_grid={}, cv=rcv, scoring='neg_root_mean_squared_error')
    gridsearcher.fit(X, y.values.ravel())
    best_br = gridsearcher.best_estimator_
    alldata_dic = {'weights': best_br.coef_, 'features': bestfeatures, 'alpha': best_br.alpha_,
                   'lambda': best_br.lambda_, 'sigma': best_br.sigma_}
    # Try to get the number of determined parameters here from the sigma ....
    # Add this dic to the mash+params_dic to make a dic within a dic
    marsh_params_dic[str(key)] = alldata_dic

    ###### Scores for simple 3 fold split ############
    cv = KFold(n_splits=3)
    scores_pred_marsh = cross_validate(baymod, X, y.values.ravel(), cv=cv, scoring=('r2', 'neg_mean_absolute_error'),
                                       n_jobs=-1)

    print('#### Bayesian Regression MODEL: 3 Fold split ')
    print(" mean RCV, and median RCV r2: ", np.mean(scores_pred_marsh['test_r2']),
          np.median(scores_pred_marsh['test_r2']))
    print(" mean RCV, and median RCV mae: ", np.mean(scores_pred_marsh['test_neg_mean_absolute_error']),
          np.median(scores_pred_marsh['test_neg_mean_absolute_error']))

    # Visualize the data
    predicted = cross_val_predict(baymod, X, y.values.ravel(), cv=cv)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.title(str(key) + " : 3 fold CV")
    plt.show()

