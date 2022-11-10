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
    'calendar_year',
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
    udf['SEC Rate (mm/yr)'] = udf['Surface Elevation Change Rate (cm/y)'] * 10
elif vertical == 'Acc_rate_fullterm (cm/y)':
    udf = udf.drop('Accretion Rate (mm/yr)', axis=1)
    #  Make sure multiplier of mass acc is in the right units
    udf['Average_Ac_cm_yr'] = udf[vertical]
    # Make sure subsidence and RSLR are in correct units
    udf['Shallow Subsidence Rate (mm/yr)'] = udf[vertical] - udf['Surface Elevation Change Rate (cm/y)']
    udf['SEC Rate (mm/yr)'] = udf['Surface Elevation Change Rate (cm/y)']
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


A = 10000  # This is the area of the study, in our case it is per site, so lets say the area is 1 m2 in cm
udf['Total Mass Accumulation (g/yr)'] = (udf['Bulk Density (g/cm3)'] * udf['Average_Ac_cm_yr']) * A  # g/cm3 * cm/yr * cm2 = g/yr
udf['Organic Mass Accumulation (g/yr)'] = (udf['Bulk Density (g/cm3)'] * udf['Average_Ac_cm_yr'] * (udf['Organic Matter (%)']/100)) * A
udf['Mineral Mass Accumulation (g/yr)'] = udf['Total Mass Accumulation (g/yr)'] - udf['Organic Mass Accumulation (g/yr)']
# Just drop the terms to be safew
udf = udf.drop([vertical, 'Average_Ac_cm_yr', 'Total Mass Accumulation (g/yr)'], axis=1)
# ########### Define outcome ########## SWITCH BETWEEN ORGANIC MASS ACC AND MINERLA MASS ACC
outcome = 'Mineral Mass Accumulation (g/yr)'
if outcome == 'Mineral Mass Accumulation (g/yr)':
    udf = udf.drop('Organic Mass Accumulation (g/yr)', axis=1)
elif outcome == 'Organic Mass Accumulation (g/yr)':
    udf = udf.drop('Mineral Mass Accumulation (g/yr)', axis=1)
else:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


# Now calcualte subsidence and RSLR
# Make the subsidence and rslr variables: using the
udf['SLR (mm/yr)'] = 2.0  # from jankowski
udf['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * udf['Latitude']) - 114.26)*-1
udf['RSLR (mm/yr)'] = udf['Shallow Subsidence Rate (mm/yr)'] + udf['Deep Subsidence Rate (mm/yr)'] + udf['SLR (mm/yr)']
udf = udf.drop('SLR (mm/yr)',  axis=1)  # obviously drop because it is the same everywhere ; only used for calc

# Drop al nans here (that way RSLR vector has same values as predictors) and remove variables related to outcome


# Try to semi-standardize variables


# Now it is feature selection time




# Lets conduct the Bayesian Ridge Regression on this dataset: do this because we can regularize w/o cross val





