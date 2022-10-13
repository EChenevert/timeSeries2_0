import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf  # is an autocorrelation plot (calculated value
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

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
# Split dataset: using dfimp
outcome = 'Accretion Rate (mm/yr)'
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, cross_val_score, GridSearchCV
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(dfimp.drop(outcome, axis=1),
                                                    dfimp[outcome], test_size=0.33, random_state=42, shuffle=True)
# Using training to train

xgbmodel = xgb.XGBRegressor()
params = {
    'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    'min_child_weight': [1, 3, 5, 7, 9, 11],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
    'max_depth': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
    'n_estimators': [25, 50, 100, 400, 500, 600, 700, 800, 900, 1000],
    'reg_lambda': [0, 0.2, 0.4, 0.6, 0.8, 1],
    # 'sunsample': [0.2, 0.4, 0.6, 0.8, 1],
    'gamma': [0, 0.2, 0.4, 0.6, 0.8, 1]
}
# 624,000 grid space
rs_model = RandomizedSearchCV(xgbmodel, param_distributions=params, n_iter=100, scoring='r2', n_jobs=-1, cv=10,
                              verbose=1)
rs_model.fit(X_train, y_train)
bestxgb = rs_model.best_estimator_

# Now use the selected features to create a model from the train data to test on the test data with repeated cv
rcv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=1)
scorestest = cross_val_score(bestxgb, X_test, y_test.values.ravel(), scoring='r2',
                         cv=rcv, n_jobs=-1)
print('### BEST XBG WHOLE DATASET ###')
print(" TEST: mean RCV, and median RCV r2: ", np.mean(scorestest), np.median(scorestest))
rcv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=1)
scorestrain = cross_val_score(bestxgb, X_train, y_train.values.ravel(), scoring='r2',
                              cv=rcv, n_jobs=-1)
print('### BEST XBG WHOLE DATASET ###')
print(" TRAIN: mean RCV, and median RCV r2: ", np.mean(scorestrain), np.median(scorestrain))

feats = ['TSS', 'NDVI', 'avg_percentflooded (%)', 'Tide_Amp (ft)',
         'avg_flooding (ft)', '90%thUpper_flooding (ft)',
         '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)',
         'Accretion Rate (mm/yr)', 'Soil Porewater Temperature (Â°C)',
         'Soil Porewater Specific Conductance (uS/cm)',
         'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)',
         'Average Height Herb (cm)']
for marsh in list(set(dfout['Community'])):
    print('######################' + str(marsh) + '#################################################')
    work = dfout[feats]
    X_train, X_test, y_train, y_test = train_test_split(work.drop(outcome, axis=1),
                                                        work[outcome], test_size=0.33, random_state=42, shuffle=True)
    ## XGBoost
    # xgbmodel = xgb.XGBRegressor()
    # params = {
    #     'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    #     'min_child_weight': [1, 3, 5, 7, 9, 11],
    #     'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
    #     'max_depth': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
    #     'n_estimators': [25, 50, 100, 400, 500, 600, 700, 800, 900, 1000],
    #     'reg_lambda': [0, 0.2, 0.4, 0.6, 0.8, 1],
    #     # 'sunsample': [0.2, 0.4, 0.6, 0.8, 1],
    #     'gamma': [0, 0.2, 0.4, 0.6, 0.8, 1]
    # }
    # # 624,000 grid space
    # rs_model = RandomizedSearchCV(xgbmodel, param_distributions=params, n_iter=100, scoring='r2', n_jobs=-1, cv=10,
    #                               verbose=1)
    # rs_model.fit(X_train, y_train)
    # bestxgb = rs_model.best_estimator_

    ## Lasso
    lassomod = linear_model.Lasso()  # sunsample=0.5, gamma=0)

    params = {
        'alpha': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                  0.95,
                  1]
    }
    # 624,000 grid space ish
    rs_model = GridSearchCV(lassomod, param_grid=params, scoring='r2', n_jobs=-1, cv=10,
                            verbose=1)
    # rs_model = RandomForestRegressor()
    rs_model.fit(X_train, y_train)
    bestlasso = rs_model.best_estimator_
    # Now use the selected features to create a model from the train data to test on the test data with repeated cv
    rcv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=1)
    scorestest = cross_val_score(bestxgb, X_test, y_test.values.ravel(), scoring='r2',
                                 cv=rcv, n_jobs=-1)
    print('### BEST XBG WHOLE DATASET ###')
    print(" TEST: mean RCV, and median RCV r2: ", np.mean(scorestest), np.median(scorestest))
    rcv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=1)
    scorestrain = cross_val_score(bestxgb, X_train, y_train.values.ravel(), scoring='r2',
                                  cv=rcv, n_jobs=-1)
    print('### BEST XBG WHOLE DATASET ###')
    print("TRAIN: mean RCV, and median RCV r2: ", np.mean(scorestrain), np.median(scorestrain))


