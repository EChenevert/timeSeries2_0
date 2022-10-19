import numpy as np
import pandas as pd
import main

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
# Concatenate
df = pd.concat([lasRates, distRiver, geefrom2020, marshElev, veg, wl, perc], axis=1, join='inner')
# Make the subsidence and rslr variables: using the
# df['Shallow Subsidence Rate (mm/yr)'] = df['Accretion Rate (mm/yr)'] -
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df['Accretion Rate (mm/yr)'], df['SEC Rate (mm/yr)'])
plt.show()


