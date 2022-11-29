from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from random import seed
import main
import pandas as pd
import numpy as np
import funcs
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, cross_val_predict, \
    cross_validate, KFold
import seaborn as sns


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
    ['Simple site', 'Distance_to_Water_m', 'Distance_to_Ocean_m']
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

# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
udf['distance_to_ocean_km'] = udf['Distance_to_Ocean_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m', 'Soil Specific Conductance (uS/cm)', 'Distance_to_Ocean_m',
                'Soil Porewater Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'tss med mg/l'})

# conduct outlier removal which drops all nans
rdf = funcs.outlierrm(udf.drop(['Community', 'Basins'], axis=1), thres=2.6)

# transformations (basically log transforamtions) --> the log actually kinda regularizes too
# rdf['log_distance_to_water_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
# rdf['log_river_width_mean_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['river_width_mean_km']]
# rdf['log_distance_to_river_km'] = [np.log10(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]
# drop the old features
# rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km', 'river_width_mean_km'], axis=1)

# Now it is feature selection time
# drop any variables related to the outcome
rdf = rdf.drop([  # IM BEING RISKY AND KEEP SHALLOW SUBSIDENCE RATE
    'Surface Elevation Change Rate (cm/y)', 'Deep Subsidence Rate (mm/yr)', 'RSLR (mm/yr)', 'SEC Rate (mm/yr)',
    # 'Shallow Subsidence Rate (mm/yr)',  # potentially encoding info about accretion
    # taking out water level features because they are not super informative
    # Putting Human in the loop
    '90th%Upper_water_level (ft NAVD88)', '10%thLower_water_level (ft NAVD88)', 'avg_water_level (ft NAVD88)',
    'std_deviation_water_level(ft NAVD88)', 'Staff Gauge (ft)', 'Soil Salinity (ppt)',
    # 'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution
    # 'Soil Porewater Temperature (°C)',
    # 'Bulk Density (g/cm3)',  'Organic Density (g/cm3)',
    # 'Soil Moisture Content (%)', 'Organic Matter (%)',
], axis=1)

# Rename some variables for better text wrapping
rdf = rdf.rename(columns={
    'Tide_Amp (ft)': 'Tide Amp (ft)',
    'avg_percentflooded (%)': 'avg percent flooded (%)',
    'Average_Marsh_Elevation (ft. NAVD88)': 'Average Marsh Elevation (ft. NAVD88)',
    'log_distance_to_water_km': 'log distance to water km',
    'log_distance_to_river_km': 'log distance to river km',
    'distance_to_ocean_km': 'distance to ocean km',
    '10%thLower_flooding (ft)': '10%thLower flooding (ft)',
    '90%thUpper_flooding (ft)': '90%thUpper flooding (ft)',
    'avg_flooding (ft)': 'avg flooding (ft)',
    'std_deviation_avg_flooding (ft)': 'std dev avg flooding (ft)'
})

# Will be using gdf because we can look into specific marsh subsets
gdf = pd.concat([rdf, udf[['Community', 'Basins']]], axis=1, join='inner')
gdf.drop(gdf.index[gdf['Community'] == 'Swamp'], inplace=True)
# drop unamed basin
gdf.drop(gdf.index[gdf['Basins'] == 'Unammed_basin'], inplace=True)
# Export gdf to file specifically for AGU data and results
# split into marsh datasets

brackdf = gdf[gdf['Community'] == 'Brackish']
saldf = gdf[gdf['Community'] == 'Saline']
freshdf = gdf[gdf['Community'] == 'Freshwater']
interdf = gdf[gdf['Community'] == 'Intermediate']
# Exclude swamp
marshdic = {'Brackish': brackdf, 'Saline': saldf, 'Freshwater': freshdf, 'Intermediate': interdf}


# EDA All Sites
# Extract the highlighted variables
# data = gdf['Bulk']
# pp = sns.pairplot(gdf[[
#         'Accretion Rate (mm/yr)', # 'Bulk Density (g/cm3)', 'Organic Matter (%)',
#        'Soil Porewater Temperature (°C)', 'Soil Porewater Salinity (ppt)',
#        'Average Height Herb (cm)', 'NDVI',
#        'Tide Amp (ft)', 'avg flooding (ft)', '90%thUpper flooding (ft)', 'std dev avg flooding (ft)',
#        'land_lost_km2', 'Community'
# ]], hue='Community')
# pp.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_allvars.png")
# plt.show()

# ## Doing pariplot for each of selected variables of backward elimination
# # all sites
# pp_all = sns.pairplot(gdf[[ 'Accretion Rate (mm/yr)', 'Soil Porewater Salinity (ppt)', 'Average Height Herb (cm)',
#                             'Average Height Dominant (cm)', 'Tide Amp (ft)', 'avg flooding (ft)',
#                             'Flood Freq (Floods/yr)']])
# pp_all.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_allsites.png")
# plt.show()
# # brackish
# pp_brack = sns.pairplot(gdf[gdf['Community'] == 'Brackish'][[ 'Accretion Rate (mm/yr)', 'Soil Porewater Salinity (ppt)', 'Average Height Herb (cm)',
#                             'Average Height Dominant (cm)']])
# pp_brack.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_brack.png")
# plt.show()
# # Freshwater log(y)
# pp_freshlogy = sns.pairplot(gdf[gdf['Community'] == 'Freshwater'][[ 'Accretion Rate (mm/yr)', 'Average Height Dominant (cm)', 'NDVI', 'tss med mg/l',
#                                   'Tide Amp (ft)', 'std dev avg flooding (ft)', 'avg percent flooded (%)']])
# pp_freshlogy.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_freshlogy.png")
# plt.show()
# # Freshwater
# pp_fresh = sns.pairplot(gdf[gdf['Community'] == 'Freshwater'][['Accretion Rate (mm/yr)', 'Average Height Dominant (cm)', 'NDVI', 'tss med mg/l',
#                                   'Tide Amp (ft)', 'std dev avg flooding (ft)', 'avg percent flooded (%)']])
# pp_fresh.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_fresh.png")
# plt.show()
# # Intermediate
# pp_inter = sns.pairplot(gdf[gdf['Community'] == 'Intermediate'][['Accretion Rate (mm/yr)', 'Average Height Dominant (cm)', 'NDVI', 'tss med mg/l',
#                                   'Tide Amp (ft)', 'std dev avg flooding (ft)', 'Flood Freq (Floods/yr)']])
# pp_inter.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_inter.png")
# plt.show()
# # Saline
# pp_inter = sns.pairplot(gdf[gdf['Community'] == 'Saline'][['Accretion Rate (mm/yr)', 'NDVI']])
# pp_inter.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\pp_inter.png")
# plt.show()

# 1st: What is the difference in the organic and mineral mass fractions
A = 10000  # This is the area of the study, in our case it is per site, so lets say the area is 1 m2 in cm
gdf['Average_Ac_cm_yr'] = gdf['Accretion Rate (mm/yr)'] * 0.10
gdf['Total Mass Accumulation (g/yr)'] = (gdf['Bulk Density (g/cm3)'] * gdf['Average_Ac_cm_yr']) * A  # g/cm3 * cm/yr * cm2 = g/yr
gdf['Organic Mass Accumulation (g/yr)'] = (gdf['Bulk Density (g/cm3)'] * gdf['Average_Ac_cm_yr'] *
                                           (gdf['Organic Matter (%)']/100)) * A
gdf['Mineral Mass Accumulation (g/yr)'] = gdf['Total Mass Accumulation (g/yr)'] - gdf['Organic Mass Accumulation (g/yr)']


# plt.figure()
# plt.title('Organic Mass Accumulation Across Marsh Types')
# sns.violinplot(data=gdf, x='Community', y='Organic Mass Accumulation (g/yr)')
# plt.show()
#
# plt.figure()
# plt.title('Mineral Mass Accumulation Across Marsh Types')
# sns.violinplot(data=gdf, x='Community', y='Mineral Mass Accumulation (g/yr)')
# plt.show()
# # No real significance difference in the trends...

# Maybe plot those weird plots of organic and mineral mass accumulation versus vertical accretion

f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
sns.boxplot(y='Mineral Mass Accumulation (g/yr)', x='Community', data=gdf,  orient='v', ax=axes[0], showfliers=False,
               palette='viridis')
sns.boxplot(y='Organic Mass Accumulation (g/yr)', x='Community', data=gdf,  orient='v', ax=axes[1], showfliers=False,
            palette='viridis')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=22)
axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=22)
plt.show()
f.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\mineral_v_organic_mass.png", dpi=300)

plt.figure()
plt.title('Organic Fraction Marsh Types')
orgviolin = sns.violinplot(data=gdf, x='Community', y='Organic Matter (%)', palette="viridis")
fig = orgviolin.get_figure()
fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\organicFrac.png", dpi=300)
plt.show()
# I guess the one significant thing here is that saline marsh has a lower organic fraction...


# 2st: Focusing on accretion and subsidence
# Make keep binary var to define highly organic from mineral soils
# Make a binary of peat or not
gdf['Mineral or Peat Soil'] = ['Peat' if val > 30 else 'Mineral' for val in gdf['Organic Matter (%)']]  # common threshold for peat... site

f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
sns.boxplot(y='Accretion Rate (mm/yr)', x='Community', data=gdf,  orient='v', ax=axes[0], showfliers=False,
               hue='Mineral or Peat Soil', # split=True,
               palette="viridis")
sns.boxplot(y='Shallow Subsidence Rate (mm/yr)', x='Community', data=gdf,  orient='v', ax=axes[1], showfliers=False,
               hue='Mineral or Peat Soil', # split=True,
               palette="viridis")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=22)
axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=22)
plt.show()
f.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\ss_and_va_perCommunity.png", dpi=300)

# Lets check out the relationship between subsidence and accretion: Definetly a positive relationship here
plt.figure()
jp = sns.jointplot(data=gdf, x='Accretion Rate (mm/yr)', y='Shallow Subsidence Rate (mm/yr)',
                   hue='Predominantly Mineral or Peat',
                   palette='rocket')
plt.show()
jp.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\ss_and_va_jp.png", dpi=300)
# Lets check the realtionship between accretion and distance to the river
g = sns.FacetGrid(gdf, col="Community", hue="Basins")
g.map(sns.scatterplot, "distance_to_river_km", "Accretion Rate (mm/yr)", alpha=.7)
g.add_legend()
plt.show()
g.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\distance_from_river_plots.png", dpi=300)


f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(1, 1, 1)
sns.histplot(data=gdf, ax=ax, stat="count", multiple="stack",
             x="Accretion Rate (mm/yr)", kde=False,
             hue="Community", #palette="pastel",
             element="bars", legend=True)
ax.set_title("Seaborn Stacked Histogram")
ax.set_xlabel("Accretion Rate (mm/yr)")
ax.set_ylabel("Count")
f.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\stacked_accretion.png", dpi=300)
plt.show()

# Maybe the source of increased accretion in terrebonne is due to reworking from land lost
plt.figure()
sns.pairplot(data=gdf[(gdf['Basins'] == 'Terrebonne') & (gdf['Community'] == 'Saline')][['Accretion Rate (mm/yr)',
                                                                                         'land_lost_km2',
                                                                                         'distance_to_water_km',
                                                                                         'Average Marsh Elevation (ft. NAVD88)',
                                                                                         'windspeed', 'Tide Amp (ft)',
                                                                                         'Soil Porewater Temperature (°C)',
                                                                                         '90%thUpper flooding (ft)',
                                                                                         'tss med mg/l', 'NDVI']])
plt.show()
# Looks like it could be due to the increase in windspeed, 90% flooding, and soil porewater temperature (??),
# cuz marsh elevation is negatively correlated to accretion here!

plt.figure()
fig = sns.scatterplot(data=gdf[(gdf['Basins'] == 'Terrebonne') & (gdf['Community'] == 'Saline')], x='90%thUpper flooding (ft)',
                      y='Accretion Rate (mm/yr)')
plt.title('Investigating Controls on Accretion in Terrebonne Basin')
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\jp_terrebonne_flooding.png", dpi=300)
plt.show()

plt.figure()
fig = sns.scatterplot(data=gdf[(gdf['Basins'] == 'Terrebonne') & (gdf['Community'] == 'Saline')], x='avg flooding (ft)',
                      y='Accretion Rate (mm/yr)')
plt.title('Investigating Controls on Accretion in Terrebonne Basin')
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\jp_terrebonne_avgflooding.png", dpi=300)
plt.show()

plt.figure()
fig = sns.jointplot(data=gdf, x='distance_to_water_km', y='Accretion Rate (mm/yr)', hue='Community')
# plt.title('Inves')
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\jp_distancewater_joint.png", dpi=300)
plt.show()

plt.figure()
fig = sns.jointplot(data=gdf, x='Bulk Density (g/cm3)', y='Accretion Rate (mm/yr)', hue='Community')
# plt.title('Inves')
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\jp_bulkDensity_joint.png", dpi=300)
plt.show()

plt.figure()
fig = sns.scatterplot(data=gdf[(gdf['Basins'] == 'Terrebonne') & (gdf['Community'] == 'Saline')], x='windspeed',
                y='Accretion Rate (mm/yr)')
plt.title('Investigatin Controls on Accretion in Terrebonne Basin')
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\jp_terrebonne_winspeed.png", dpi=300)
plt.show()

# Lets investigate NDVI: particularly for freshwater and saline marshes since they got opposite effects
plt.figure()
jpndvi = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='NDVI', y='Accretion Rate (mm/yr)', hue='Community', palette="rocket")
jpndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\ndvi_freshandsal.png", dpi=300)
plt.show()

plt.figure()
salflood = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='Soil Porewater Salinity (ppt)', y='90%thUpper flooding (ft)', hue='Community', palette="rocket")
salflood.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\salflood.png", dpi=300)
plt.show()

plt.figure()
jpndvi = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='Soil Porewater Salinity (ppt)', y='Accretion Rate (mm/yr)', hue='Community', palette="rocket")
jpndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\salinity_freshandsal.png", dpi=300)
plt.show()

plt.figure()
domveg = sns.jointplot(data=gdf, x='Average Height Dominant (cm)', y='Accretion Rate (mm/yr)', hue='Community')
domveg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\dominantVeg.png", dpi=300)
plt.show()

plt.figure()
herbveg = sns.jointplot(data=gdf, x='Average Height Herb (cm)', y='Accretion Rate (mm/yr)',
              hue='Community')
herbveg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\herbVeg.png", dpi=300)
plt.show()

plt.figure()
domherbveg = sns.jointplot(data=gdf, x='Average Height Herb (cm)',
                           y='Average Height Dominant (cm)',
                            hue='Community')
domherbveg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\domherbVeg.png", dpi=300)
plt.show()

plt.figure()
herblat = sns.jointplot(data=gdf, x='Average Height Herb (cm)',
                           y='Latitude',
                            hue='Community')
herblat.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\herblat.png", dpi=300)
plt.show()

plt.figure()
herblat = sns.jointplot(data=gdf, x='Average Height Herb (cm)',
                           y='Latitude',
                            hue='Community')
herblat.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\herblat.png", dpi=300)
plt.show()

plt.figure()
tide = sns.jointplot(data=gdf, x='Tide Amp (ft)', y='Accretion Rate (mm/yr)',
              hue='Community')
tide.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\tide.png", dpi=300)
plt.show()

plt.figure()
floodndvi = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='avg flooding (ft)', y='NDVI', hue='Community', palette="rocket")
floodndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\floodndvi_freshandsal.png", dpi=300)
plt.show()


plt.figure()
upperfloodndvi = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='90%thUpper flooding (ft)', y='NDVI', hue='Community', palette="rocket")
upperfloodndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\upperfloodndvi_freshandsal.png", dpi=300)
plt.show()

plt.figure()
vaorg = sns.jointplot(data=gdf,
                x='Organic Matter (%)', y='Accretion Rate (mm/yr)', hue='Community')
vaorg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\vaorg.png", dpi=300)
plt.show()

plt.figure()
vaorg = sns.jointplot(data=gdf,
                x='Organic Matter (%)', y='NDVI', hue='Community')
vaorg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\vaorg.png", dpi=300)
plt.show()

# Looking into soil temperature
plt.figure()
temporg = sns.jointplot(data=gdf,
                x='Soil Porewater Temperature (°C)', y='Organic Matter (%)', hue='Community')
temporg.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\temporg.png", dpi=300)
plt.show()

plt.figure()
tempva = sns.jointplot(data=gdf,
                x='Soil Porewater Temperature (°C)', y='Accretion Rate (mm/yr)', hue='Community')
tempva.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\tempva.png", dpi=300)
plt.show()

plt.figure()
templat = sns.jointplot(data=gdf,
                y='Soil Porewater Temperature (°C)', x='Latitude', hue='Community')
templat.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\templat.png", dpi=300)
plt.show()

plt.figure()
tempbulk = sns.jointplot(data=gdf,
                y='Soil Porewater Temperature (°C)', x='Bulk Density (g/cm3)', hue='Community')
tempbulk.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\tempbulk.png", dpi=300)
plt.show()

plt.figure()
tideflood = sns.jointplot(data=gdf,
                y='Tide Amp (ft)', x='avg flooding (ft)', hue='Community')
tideflood.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\tideflood.png", dpi=300)
plt.show()

plt.figure()
orgndvi = sns.jointplot(data=gdf,
                y='Organic Matter (%)', x='NDVI', hue='Community')
orgndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\orgndvi.png", dpi=300)
plt.show()

plt.figure()
salfresh_orgndvi = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                y='Organic Matter (%)', x='NDVI', hue='Community', palette='rocket')
salfresh_orgndvi.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\salfresh_orgndvi.png", dpi=300)
plt.show()

plt.figure()
templatsalfresh = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='Soil Porewater Temperature (°C)', y='Latitude', hue='Community', palette='rocket')
templatsalfresh.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\templatsalfresh.png", dpi=300)
plt.show()

plt.figure()
tempvasalfresh = sns.jointplot(data=gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Freshwater')],
                x='Soil Porewater Temperature (°C)', y='Accretion Rate (mm/yr)', hue='Community', palette='rocket')
tempvasalfresh.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\tempvasalfresh.png", dpi=300)
plt.show()

plt.figure()
latvasalfresh = sns.jointplot(data=gdf,
                x='Latitude', y='Accretion Rate (mm/yr)', hue='Community')
latvasalfresh.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\latvasalfresh.png", dpi=300)
plt.show()

plt.figure()
oceantemp = sns.jointplot(data=gdf,
                x='distance to ocean km', y='Soil Porewater Temperature (°C)', hue='Community')
oceantemp.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_forscaledXY\\oceantemp.png", dpi=300)
plt.show()
