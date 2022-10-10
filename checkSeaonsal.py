import main
import pandas as pd
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
# Condense into just yearly because i only get yearly tide amp measurements easily
avgSeasons['Year (yyyy)'] = avgSeasons['Year (yyyy)'].astype(str)
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
wldf['calendar_year'] = wldf['calendar_year'].astype(str)
reWL = wldf.groupby(['Simple site', 'calendar_year']).median()
ccdf = pd.concat([reWL, impYrsDf], axis=1)

# Add the percent time flooded variable:
pfl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\11968_PercentFlooded_CalendarYearTimeSeries\11968.csv",
                  encoding="unicode_escape")
pfltest = pfl[pfl["Station_ID"].str.contains("H") == True]
pfltest['Simple site'] = [i[:8] for i in pfltest['Station_ID']]
pfldf = pfltest[['Simple site', 'Year', 'avg_percentflooded (%)']]
pfldf['Year'] = pfldf['Year'].astype(str)
rePFL = pfldf.groupby(['Simple site', 'Year']).median()
pwccdf = pd.concat([rePFL, ccdf], axis=1)

# add the remote sensing data (YEARLY):
# NDVI
ndviTS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\table_demo_NDVI_CRMS3.csv", encoding='unicode_escape')
ndviTS['Year'] = [i[:4] for i in ndviTS['system:index']]
newNDVIts = ndviTS.drop('system:index', axis=1)
dicNDVI = {'Year': [], 'Simple site': [], 'NDVI': []}

listSites = list(newNDVIts.columns.values)
listSites.remove('.geo')
listSites.remove('imageId')
listSites.remove('Year')
for col in listSites:
    diffdf = newNDVIts[['Year', col]]
    diffdf['Simple site'] = col
    dicNDVI['Year'] = dicNDVI['Year'] + list(diffdf['Year'])
    dicNDVI['Simple site'] = dicNDVI['Simple site'] + list(diffdf['Simple site'])
    dicNDVI['NDVI'] = dicNDVI['NDVI'] + list(diffdf[col])

dfndvi = pd.DataFrame(dicNDVI)
dfndvigb = dfndvi.groupby(['Simple site', 'Year']).median()

# TSS
tssTS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\table_demo_TSS_CRMS.csv", encoding='unicode_escape')
tssTS['Year'] = [i[:4] for i in tssTS['system:index']]
newTSSts = tssTS.drop('system:index', axis=1)
dicTSS = {'Year': [], 'Simple site': [], 'TSS': []}

tssSites = list(newTSSts.columns.values)
tssSites.remove('.geo')
tssSites.remove('imageId')
tssSites.remove('Year')
for col in tssSites:
    diffdf = newTSSts[['Year', col]]
    diffdf['Simple site'] = col
    dicTSS['Year'] = dicTSS['Year'] + list(diffdf['Year'])
    dicTSS['Simple site'] = dicTSS['Simple site'] + list(diffdf['Simple site'])
    dicTSS['TSS'] = dicTSS['TSS'] + list(diffdf[col])

dftss = pd.DataFrame(dicTSS)
dftssgb = dftss.groupby(['Simple site', 'Year']).median()

# Combine the datasets
rsdf = pd.concat([dftssgb, dfndvigb], join='inner', axis=1)
alldf = pd.concat([rsdf, pwccdf], join='inner', axis=1).reset_index()

# Attach the correct basin and marsh type to each specific site.
marshComRef = avgBysite[['Simple site', 'Community']]
dicMarshSite = dict(zip(marshComRef['Simple site'], marshComRef['Community']))
alldf['Community'] = [dicMarshSite[site] for site in alldf['Simple site']]

basinComRef = avgBysite[['Simple site', 'Basins']]
dicBasinSite = dict(zip(basinComRef['Simple site'], basinComRef['Basins']))
alldf['Basins'] = [dicBasinSite[site] for site in alldf['Simple site']]

alldf.to_csv("D:\\Etienne\\fall2022\\CRMS_data\\timeseries_CRMS.csv")
