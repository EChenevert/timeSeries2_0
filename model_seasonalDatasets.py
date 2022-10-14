import pandas as pd
import main
import funcs

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
    'Delta time (days)', 'Delta Time (decimal_years)', 'Staff Gauge (ft)', 'Measurement Depth (ft)',
    'Soil Porewater Temperature (°C)', 'Direction (Collar Number)', 'Direction (Compass Degrees)',
    'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)'
], axis=1)
summerDF = summerDF.drop([
    'Average Accretion (mm)',  # may wanna check this oucome outcome
    'Year (yyyy)', 'Season', 'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)',
    'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)', 'Latitude', 'Longitude', 'Month (mm)',
    'Delta time (days)', 'Delta Time (decimal_years)', 'Soil Moisture Content (%)', 'Bulk Density (g/cm3)',
    'Organic Matter (%)', 'Wet Volume (cm3)', 'Organic Density (g/cm3)', 'Staff Gauge (ft)', 'Measurement Depth (ft)',
    'Soil Porewater Temperature (°C)', 'Direction (Collar Number)', 'Direction (Compass Degrees)',
    'Pin Number', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)'
], axis=1)
