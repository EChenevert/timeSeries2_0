import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf  # is an autocorrelation plot (calculated value
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\\Etienne\\fall2022\\CRMS_data\\timeseries_CRMS.csv", encoding="unicode_escape")
# First clea nthe daatset to the point of EDA
df = df.drop(['Unnamed: 0', 'Season', 'Staff Gauge (ft)'], axis=1)  # drop uninformative columns
df = df.dropna(subset='Accretion Rate (mm/yr)')

def outlierrm(df, thres=3):
    """Dont put in long lats in here! Need Year and Site name lol"""
    df = df.dropna().set_index(['Simple site', 'level_1'])
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


dfout = outlierrm(df).reset_index()
# dfout = df
# First visualize through time
sns.regplot(data=dfout, x='level_1', y='Accretion Rate (mm/yr)')
plt.show()
# Fit linear model: if time can predict accretion, then they are NOT independent (significant p-val)
import statsmodels.api as sm

X = sm.add_constant(dfout['level_1'])
res_modAll = sm.OLS(dfout['Accretion Rate (mm/yr)'], X).fit()
print(res_modAll.summary())

# Check temporal dependence:
# Note: Limiting Lags to 50
data = dfout[['level_1', 'Accretion Rate (mm/yr)']].set_index(['level_1']).sort_index()
plot_acf(x=data, lags=50)
# Show the AR as a plot
plt.show()  # anything within the 95% confidence interval (shaded, defualt alpha=0.05) has no significant correlation

# for marsh in list(set(dfout['Community'])):
#     dfmarsh = dfout[dfout['Community'] == marsh]
#     # dfmarsh = dfmarsh.reset_index().drop(0)
#     sns.regplot(data=dfmarsh, x='level_1', y='Accretion Rate (mm/yr)')
#     plt.title(marsh)
#     plt.show()
#     # Fit linear model: if time can predict accretion, then they are NOT independent (significant p-val)
#     import statsmodels.api as sm
#     X = sm.add_constant(dfmarsh['level_1'])
#     res_modAll = sm.OLS(dfmarsh['Accretion Rate (mm/yr)'], X).fit()
#     print(res_modAll.summary())
#
#     # Check temporal dependence:
#     # Note: Limiting Lags to 50
#     data = dfmarsh[['level_1', 'Accretion Rate (mm/yr)']].set_index(['level_1']).sort_index()
#
#     plot_acf(x=data, lags=50)
#     # Show the AR as a plot
#     plt.title(marsh)
#     plt.show()  # anything within the 95% confidence interval (shaded, defualt alpha=0.05) has no significant correlation

# # Doing autocorrelation for every fucking site
# for site in list(set(dfout['Simple site'])):
#     dfsite = dfout[dfout['Simple site'] == site]
#     if len(dfsite) < 9:
#         pass
#     else:
#         sns.regplot(data=dfsite, x='level_1', y='Accretion Rate (mm/yr)')
#         plt.title(site)
#         plt.show()
#         # Fit linear model: if time can predict accretion, then they are NOT independent (significant p-val)
#         import statsmodels.api as sm
#
#         X = sm.add_constant(dfsite['level_1'])
#         res_modAll = sm.OLS(dfsite['Accretion Rate (mm/yr)'], X).fit()
#         print(res_modAll.summary())
#
#         # Check temporal dependence:
#         # Note: Limiting Lags to 50
#         data = dfsite[['level_1', 'Accretion Rate (mm/yr)']].set_index(['level_1']).sort_index()
#
#         plot_acf(x=data, lags=2)
#         # Show the AR as a plot
#         plt.title(site)
#         plt.show()  # anything within the 95% confidence interval (shaded, defualt alpha=0.05) has no significant correlation


### Visualize the kde distributions
dfimp = dfout[['TSS', 'NDVI', 'avg_percentflooded (%)', 'Tide_Amp (ft)', 'avg_flooding (ft)', '90%thUpper_flooding (ft)',
       '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)', 'Accretion Rate (mm/yr)',
       'Soil Porewater Temperature (Â°C)', 'Soil Porewater Specific Conductance (uS/cm)',
       'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Average Height Herb (cm)']]
for col in dfimp.columns:
    sns.kdeplot(data=dfout, x=col, hue='Community')
    plt.title(col)
    plt.show()
    ## For the most part they all hang around the same
    # sns.boxplot(data=dfout, x='Community', y=col, showfliers=False)
    # plt.title(col)
    # plt.show()

### Do a lil XGBoost on the whole dataset...
#...... DO THESEEEEE >>>>>>>>>>>>>>>

# V


# First

