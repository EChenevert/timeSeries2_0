from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from random import seed

from sklearn.preprocessing import StandardScaler

import main
import pandas as pd
import numpy as np
import funcs
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, cross_val_predict, \
    cross_validate, KFold
import seaborn as sns
import matplotlib


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
    ['Simple site', 'Distance_to_Water_m']  # 'Distance_to_Ocean_m'
].set_index('Simple site')
floodfreq = pd.read_csv(r"D:\\Etienne\\fall2022\\agu_data\\floodFrequencySitePerYear.csv", encoding="unicode_escape")[[
    'Simple site', 'Flood Freq (Floods/yr)'
]].set_index('Simple site')

# Concatenate
df = pd.concat([bysite, distRiver, nearWater, gee, jrc, marshElev, wl, perc, SEC, acc, floodfreq], axis=1, join='outer')

# Now clean the columns
# First delete columns that are more than 1/2 nans
tdf = df.dropna(thresh=df.shape[0]*0.5, how='all', axis=1)
# Drop uninformative features
udf = tdf.drop([
    'Year (yyyy)', 'Accretion Measurement 1 (mm)', 'Year',
    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    'Accretion Measurement 4 (mm)',
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
    udf = udf.drop(['SLR (mm/yr)'],
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
    udf = udf.drop(['SLR (mm/yr)'],
                   axis=1)  # obviously drop because it is the same everywhere ; only used for calc
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

####### Define outcome as vertical component
outcome = vertical

udf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\AGU_dataset_noOutlierRm.csv")
# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
# udf['distance_to_ocean_km'] = udf['Distance_to_Ocean_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', #  'Distance_to_Ocean_m',
                'Soil Specific Conductance (uS/cm)',
                'Soil Porewater Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'TSS (mg/l)'})

# Delete the swamp sites and unammed basin
udf.drop(udf.index[udf['Community'] == 'Swamp'], inplace=True)
# udf.drop(udf.index[udf['Basins'] == 'Unammed_basin'], inplace=True)
udf = udf.drop('Basins', axis=1)
# conduct outlier removal which drops all nans
rdf = funcs.outlierrm(udf.drop(['Community', 'Latitude', 'Longitude'], axis=1), thres=3)

# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
rdf['log_river_width_mean_km'] = [np.log(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
rdf['log_distance_to_river_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
# rdf['log_distance_to_ocean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_ocean_km']]
# rdf['Average Height Dominant (mm)'] = rdf['Average Height Dominant (cm)'] * 10
# rdf['Average Height Herb (mm)'] = rdf['Average Height Herb (cm)'] * 10
# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km', 'river_width_mean_km'], axis=1)  # 'distance_to_ocean_km'
# Now it is feature selection time
# drop any variables related to the outcome
rdf = rdf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
    'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    # taking out water level features because they are not super informative
    # Putting Human in the loop
    '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
    'std_deviation_water_level(ft NAVD88)', 'Staff Gauge (ft)', 'Soil Salinity (ppt)',
    'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution
    # Delete the dominant herb cuz of rendundancy with dominant veg
    'Average Height Herb (cm)',
    # 'tss med mg/l',  # cuz idk if i trust calc..... eh
    # other weird ones
    'Soil Porewater Temperature (°C)',
    'Average_Marsh_Elevation (ft. NAVD88)',
    'Bulk Density (g/cm3)',  'Organic Density (g/cm3)',
    'Soil Moisture Content (%)',  'Organic Matter (%)',  # do not use organic matter because it has a negative relationship, hard for me to interpret --> i think just picks up the bulk density relationship. Or relationship that sites with higher organic matter content tend to have less accretion
    'land_lost_km2'
], axis=1)

# Rename some variables for better text wrapping
rdf = rdf.rename(columns={
    'Tide_Amp (ft)': 'Tide Amp (ft)',
    'avg_percentflooded (%)': 'Avg. Time Flooded (%)',
    'windspeed': 'Windspeed',
    # 'log_distance_to_ocean_km': 'log distance to ocean km',
    # 'Average_Marsh_Elevation (ft. NAVD88)': 'Average Marsh Elevation (ft. NAVD88)',
    'log_distance_to_water_km': 'Log Distance to Water (km)',
    'log_distance_to_river_km': 'Log Distance to River (km)',
    '10%thLower_flooding (ft)': '10th Percentile of Flooding (ft)',
    '90%thUpper_flooding (ft)': '90th Percentile of Flooding (ft)',
    'avg_flooding (ft)': 'Avg. Flooding (ft)',
    'std_deviation_avg_flooding (ft)': 'Std. Deviation of Flooding (ft)'
})

gdf = pd.concat([rdf, udf[['Community', 'Longitude', 'Latitude']]], axis=1, join='inner')
# Export gdf to file specifically for AGU data and results
gdf.to_csv("D:\\Etienne\\fall2022\\agu_data\\results\\AGU_dataset.csv")

# split into marsh datasets

brackdf = gdf[gdf['Community'] == 'Brackish']
saldf = gdf[gdf['Community'] == 'Saline']
freshdf = gdf[gdf['Community'] == 'Freshwater']
interdf = gdf[gdf['Community'] == 'Intermediate']
combined = gdf[(gdf['Community'] == 'Intermediate') | (gdf['Community'] == 'Brackish')]
# Exclude swamp
marshdic = {'All': gdf, 'Brackish': brackdf, 'Saline': saldf, 'Freshwater': freshdf, 'Intermediate': interdf,
            'Intermediate and Brackish': combined}


hold_marsh_weights = {}
hold_unscaled_weights = {}
hold_intercept = {}
hold_marsh_regularizors = {}
hold_marsh_weight_certainty = {}
hold_prediction_certainty = {}

for key in marshdic:
    print(key)
    mdf = marshdic[key]  # .drop('Community', axis=1)
    # It is preshuffled so i do not think ordering will be a problem
    t = np.log10(mdf[outcome].reset_index().drop('index', axis=1))
    phi = mdf.drop([outcome, 'Community', 'Latitude', 'Longitude'], axis=1).reset_index().drop('index', axis=1)
    # Scale: because I want feature importances
    scalar_Xmarsh = StandardScaler()
    predictors_scaled = pd.DataFrame(scalar_Xmarsh.fit_transform(phi), columns=phi.columns.values)
    # # NOTE: I do feature selection using whole dataset because I want to know the imprtant features rather than making a generalizable model
    # mlr = linear_model.LinearRegression()
    # br = linear_model.BayesianRidge(fit_intercept=False)
    #
    # feature_selector = ExhaustiveFeatureSelector(br,
    #                                                  min_features=1,
    #                                                  max_features=3,
    #                                                  # I should only use 5 features (15 takes waaaaay too long)
    #                                                  scoring='neg_root_mean_squared_error',
    #                                                  # print_progress=True,
    #                                                  cv=3)  # 5 fold cross-validation
    #
    # efsmlr = feature_selector.fit(predictors_scaled, target_scaled.values.ravel())  # these are not scaled... to reduce data leakage
    #
    # print('Best CV r2 score: %.2f' % efsmlr.best_score_)
    # print('Best subset (indices):', efsmlr.best_idx_)
    # print('Best subset (corresponding names):', efsmlr.best_feature_names_)
    #
    # bestfeaturesM = list(efsmlr.best_feature_names_)

    bestfeaturesM = funcs.backward_elimination(predictors_scaled, t.values.ravel(), num_feats=20,
                                               significance_level=0.05)

    # Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val
    #### NOTE: I should do separate tests to determine which split of the data is optimal ######
    # first split data set into test train
    from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

    X, y = predictors_scaled[bestfeaturesM], t

    baymod = linear_model.BayesianRidge(fit_intercept=True)

    results_dict = funcs.log10_cv_results_and_plot(baymod, bestfeaturesM, phi, X, y, {'cmap': 'YlOrRd', 'line': "r--"}, str(key))

    hold_marsh_weights[key] = results_dict["Scaled Weights"]
    hold_unscaled_weights[key] = results_dict["Unscaled Weights"]
    hold_marsh_regularizors[key] = results_dict["Scaled regularizors"]
    hold_marsh_weight_certainty[key] = results_dict["# Well Determined Weights"]
    hold_prediction_certainty[key] = results_dict["Standard Deviations of Predictions"]
    hold_intercept[key] = results_dict["Unscaled Intercepts"]

# Make a colormap so all each weight will have a specific color
colormap = {
'Soil Porewater Salinity (ppt)': '#DD8A8A',
'Average Height Dominant (cm)': '#137111',
'NDVI': '#0AFF06',
'TSS (mg/l)': '#8E6C02',
'Windspeed': '#70ECE3',
'Tide Amp (ft)': '#434F93',
'Avg. Flooding (ft)': '#087AFA',
'90th Percentile of Flooding (ft)': '#D000E1',
'10th Percentile of Flooding (ft)': '#73ABAE',
'Std. Deviation of Flooding (ft)': '#DE5100',
'Avg. Time Flooded (%)': '#970CBD',
'Flood Freq (Floods/yr)': '#EB0000',
'Log Distance to Water (km)': '#442929',
'Log Distance to River (km)': '#045F38',
}

for key in hold_marsh_weights:
    d = pd.DataFrame(hold_marsh_weights[key].mean().reset_index()).rename(columns={0: 'Means'})
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7, 7))
    # my_cmap = plt.get_cmap("cool")
    # ax.bar(list(d['index']), list(d['Means']), color='Blue')
    ax.set_title(str(key) + " Sites")
    # sns.barplot(data=hold_marsh_weights[key], palette="Blues")
    palette_ls = []
    for weight in d['index']:
        palette_ls.append(colormap[weight])
    sns.barplot(list(d['index']), list(d['Means']), palette=palette_ls)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\" + str(key) +
                "_scaledX_nolog_boxplot_human.eps", format='eps',
                dpi=500,
                bbox_inches='tight')
    plt.show()

# Plot the distribution of weight parameters for the marsh runs
for key in hold_unscaled_weights:
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7, 7))
    # matplotlib.rcParams['pdf.fonttype'] = 42
    ax.set_title(str(key) + " Sites")
    if key != 'Saline':
        ax.axhline(0, ls='--')
    palette_ls = []
    for weight in hold_unscaled_weights[key].keys():
        palette_ls.append(colormap[weight])
    boxplot = sns.boxplot(data=hold_unscaled_weights[key], notch=True, showfliers=False, palette=palette_ls)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\" + str(
        key) + "_unscaledWeights_nolog_boxplot_human.eps", format='eps',
                dpi=500,
                bbox_inches='tight')
    plt.show()


# Plot the distribution of the eff_reg parameter for each run
eff_reg_df = pd.DataFrame(hold_marsh_regularizors)
sns.set_theme(style='white', font_scale=1)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Learned Effective Regularization Parameters')
sns.boxplot(data=eff_reg_df, notch=True, showfliers=False, palette="YlOrBr")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\regularization_scaledX_nolog_boxplot_human.eps",
            format='eps',
            dpi=500,
            bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of parameters for each run
certainty_df = pd.DataFrame(hold_marsh_weight_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Calculated Number of Well Determined Parameters')
sns.boxplot(data=certainty_df, notch=True, showfliers=False, palette="Blues")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\certainty_scaledX_nolog_boxplot_human.eps",
            format='eps',
            dpi=500,
            bbox_inches='tight')
plt.show()



# Plot the distribution calculated intercepts
intercept_df = pd.DataFrame(hold_intercept)
sns.set_theme(style='white', rc={'figure.dpi': 147}, font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Intercepts [Unscaled]:')
ax.axhline(0, ls='--')
sns.boxplot(data=intercept_df, notch=True, showfliers=False, palette="coolwarm")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\intercepts_nolog_boxplot_human.eps", dpi=500,
            format='eps',
            bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of predictions for each run
pred_certainty_df = pd.DataFrame(hold_prediction_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Bayesian Uncertainty in Predictions')
sns.boxplot(data=pred_certainty_df, notch=True, showfliers=False, palette="Reds")
funcs.wrap_labels(ax, 10)
fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\pred_certainty_scaledX_nolog_boxplot_human.eps",
            dpi=500, format='eps',
            bbox_inches='tight')
plt.show()

# Following https://christophm.github.io/interpretable-ml-book/limo.html for individual feature importances
# Want to show points for the 10th, 25th, 50th, 75th, 90th poins of outcome and their feature effects
