import main
import pandas as pd
import numpy as np

dataframes = main.load_data()
avgSeasons = main.average_byyear_bysite_seasonal(dataframes)

springWinterTS = avgSeasons[avgSeasons['Season'] == 2]
summerTS = avgSeasons[avgSeasons['Season'] == 1]
# Condense into just yearly because i only get yearly tide amp measurements easily
yearsDf = avgSeasons.groupby(['Simple site', 'Year (yyyy)']).median()
impYrsDf = yearsDf[['Season', 'Average Accretion (mm)', 'Accretion Rate (mm/yr)', 'Staff Gauge (ft)',
                    'Soil Porewater Temperature (°C)', 'Soil Porewater Specific Conductance (uS/cm)',
                    'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Average Height Herb (cm)']]
# Clean dataframe from relatively empty columns : Drop rows without accretion values
# Remove columns that are more than 50% complete
cleanSW = springWinterTS.dropna(thresh=springWinterTS.shape[0]*0.5, how='all', axis=1)
cleanS = summerTS.dropna(thresh=summerTS.shape[0]*0.5, how='all', axis=1)

# # Only get the important data
# impSW = cleanSW[['Simple site', 'Year (yyyy)', 'Season', 'Average Accretion (mm)', 'Accretion Rate (mm/yr)', 'Basins',
#                  'Staff Gauge (ft)', 'Soil Porewater Temperature (°C)', 'Soil Porewater Specific Conductance (uS/cm)',
#                  'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Average Height Herb (cm)',
#                  'Community']]
# impS = cleanS[['Simple site', 'Year (yyyy)', 'Season', 'Average Accretion (mm)', 'Accretion Rate (mm/yr)', 'Basins',
#                'Staff Gauge (ft)', 'Soil Porewater Temperature (°C)', 'Soil Porewater Specific Conductance (uS/cm)',
#                'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Average Height Herb (cm)',
#                'Community']]

# Expand on the site and year average dataset
wl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\11966_WaterLevelRange_CalendarYearTimeSeriesAll\11966.csv",
                 encoding='unicode_escape')
# Make the site name simple: Only use the CRMS0000 - H sites tho this time ... should be more consistent
wltest = wl[wl["Station_ID"].str.contains("H") == True]
# Now instill the simple site
wltest['Simple site'] = [i[:8] for i in wltest['Station_ID']]
# Only take relevant variables and set index to simple site and year for concatenation with other df
wldf = wltest[['Tide_Amp (ft)', 'calendar_year', 'avg_flooding (ft)', '90%thUpper_flooding (ft)',
               '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)', 'Simple site']]
reWL = wldf.groupby(['Simple site', 'calendar_year']).median()
ccdf = pd.concat([reWL, impYrsDf], axis=1)

# Add the percent time flooded variable:

# add the remote sensing data:
