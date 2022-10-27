import numpy as np
import pandas as pd
import main
from scipy import stats


data = main.load_data()
yearly = main.average_byyear_bysite_seasonal(data)
lasRates = yearly[yearly['Year (yyyy)'] > 2019].groupby('Simple site').median()  # [yearly['Year (yyyy)'] > 2019]

perc = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12006_PercentFlooded_CalendarYear\12006.csv",
                   encoding="unicode escape")
perc['Simple site'] = [i[:8] for i in perc['Station_ID']]
perc = perc.groupby('Simple site').median()
wl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12007_WaterLevelRange_CalendarYear\12007.csv",
                 encoding="unicode escape")
wl['Simple site'] = [i[:8] for i in wl['Station_ID']]
wl = wl.groupby('Simple site').median()

veg = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12008_VegPercentCover\12008.csv",
                  encoding="unicode escape").groupby('Site_ID').median()
marshElev = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12009_Survey_Marsh_Elevation\12009_Survey_Marsh_Elevation.csv",
                        encoding="unicode escape").groupby('SiteId').median()
geefrom2020 = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_GEE90percfrom2020.csv",
                          encoding="unicode escape")[['Simple_sit', 'NDVI', 'tss_med', 'windspeed']]\
    .groupby('Simple_sit').median()
distRiver = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv",
                        encoding="unicode escape")[['Field1', 'distance_to_river_m']].groupby('Field1').median()
SEC = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\12017_SurfaceElevation_ChangeRate\12017.csv",
                  encoding="unicode escape")
SEC['Simple site'] = [i[:8] for i in SEC['Station_ID']]
SEC = SEC.groupby('Simple site').median()
# Concatenate
df = pd.concat([lasRates, distRiver, geefrom2020, marshElev, veg, wl, perc, SEC], axis=1, join='inner')

# Make the subsidence and rslr variables: using the
df['Shallow Subsidence Rate (mm/yr)'] = df['Accretion Rate (mm/yr)'] - df['Surface Elevation Change Rate (cm/y)']*10
df['SEC Rate (mm/yr)'] = df['Surface Elevation Change Rate (cm/y)']*10
df['SLR (mm/yr)'] = 2.0  # from jankowski
df['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * df['Latitude']) - 114.26)*-1
df['RSLR (mm/yr)'] = df['Shallow Subsidence Rate (mm/yr)'] + df['Deep Subsidence Rate (mm/yr)'] + df['SLR (mm/yr)']

# Clean dataset
df = df.dropna(subset='Accretion Rate (mm/yr)')
df = df.dropna(thresh=df.shape[0]*0.9, how='all', axis=1)


def outlierrm(df, thres=3):
    """Dont put in long lats in here! Need Year and Site name lol"""
    df = df.dropna()
    switch = False
    if 'Basins' in df.columns.values or 'Community' in df.columns.values:
        print('True')
        switch = True
        holdstrings = df[['Basins', 'Community']]
        df = df.drop(['Basins', 'Community'], axis=1)
    df = df.apply(pd.to_numeric)
    length = len(df.columns.values)
    for col in df.columns.values:
        df[col + "_z"] = stats.zscore(df[col])
    for col in df.columns.values[length:]:
        df = df[np.abs(df[col]) < thres]
    df = df.drop(df.columns.values[length:], axis=1)
    if switch:
        df = pd.concat([df, holdstrings], join='inner', axis=1)
    return df



dfi = df[[
    'RSLR (mm/yr)', 'Accretion Rate (mm/yr)', 'avg_flooding (ft)', '90%thUpper_flooding (ft)',
    '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)', 'avg_percentflooded (%)', 'distance_to_river_m',
    'NDVI', 'windspeed', 'Tide_Amp (ft)'
]]

dfi = outlierrm(dfi, thres=1.3)

dfi.to_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_dfi.csv")
