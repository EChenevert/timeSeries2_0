import pandas as pd
import numpy as np
import csv


# Functions used to load data
def organized_iteryears(date_col_name, df):
    ''' This method creates a data column that indicates what year the sample
    (or data) was collected or observed'''
    datetimeArr = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
    years = datetimeArr.dt.year
    return years


def organized_itermons(date_col_name, df):
    '''This is an iterdates method for the hydro data, which is logged into
    the csv more cleaning. This increases speed
    @params:
        date_col_names = is the name of the date column
        df = is the dataframe the date column is in'''
    datetimeArr = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
    months = datetimeArr.dt.month
    return months


def add_basins(df, basin_str, ls_crms):
    for i in range(len(ls_crms)):
        indices = [k for k, x in enumerate(df['Simple site']) if x == ls_crms[i]]
        for j in range(len(indices)):
            df['Basins'][indices[j]] = basin_str
    return df


def add_avgAccretion(accdf):
    avg_accretion = (accdf['Accretion Measurement 1 (mm)'] + accdf['Accretion Measurement 2 (mm)'] +
                     accdf['Accretion Measurement 3 (mm)'] + accdf['Accretion Measurement 4 (mm)']) / 4
    avg_accretion = pd.DataFrame(avg_accretion, columns=['Average Accretion (mm)'], index=accdf.index.values)
    newdf = pd.concat([accdf, avg_accretion], axis=1)
    return newdf


def add_accretionRate(accdf):
    accdf['Average Accretion (mm)'] = (accdf['Accretion Measurement 1 (mm)'] + accdf['Accretion Measurement 2 (mm)'] +
                                       accdf['Accretion Measurement 3 (mm)'] +
                                       accdf['Accretion Measurement 4 (mm)']) / 4
    accdf['Sample Date (mm/dd/yyyy)'] = pd.to_datetime(accdf['Sample Date (mm/dd/yyyy)'],
                                                       format='%m/%d/%Y')

    accdf['Establishment Date (mm/dd/yyyy)'] = pd.to_datetime(accdf['Establishment Date (mm/dd/yyyy)'],
                                                              format='%m/%d/%Y')

    accdf['Delta time (days)'] = accdf['Sample Date (mm/dd/yyyy)'] - \
                                 accdf['Establishment Date (mm/dd/yyyy)']

    accdf['Delta time (days)'] = accdf['Delta time (days)'].dt.days
    accdf['Delta Time (decimal_years)'] = accdf['Delta time (days)'] / 365
    accdf['Accretion Rate (mm/yr)'] = accdf['Average Accretion (mm)'] / accdf['Delta Time (decimal_years)']

    return accdf


def convert_str(string):
    '''Converts a string into a list'''
    ls = list(string.split(', '))
    return ls


def load_data():

    '''This loads all the crms data currently in the data folder of this package'''

    soil_properties = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Soil_Properties\CRMS_Soil_Properties.csv", encoding='unicode escape')
    # hourly_hydro = pd.read_csv(r"C:\Users\etachen\Documents\PyCharmProjs\datasetsCRMS\main\data\CRMS_Continuous_Hydrographic.csv", encoding='unicode escape')
    monthly_hydro = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Discrete_Hydrographic\CRMS_Discrete_Hydrographic.csv", encoding='unicode escape')
    marsh_vegetation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Marsh_Vegetation\CRMS_Marsh_Vegetation.csv", encoding='unicode escape')
    forest_vegetation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Forest_Vegetation\CRMS_Forest_Vegetation.csv", encoding='unicode escape')
    accretion = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Accretion\CRMS_Accretion.csv", encoding='unicode escape')
    biomass = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Biomass\CRMS_Biomass.csv", encoding='unicode escape')
    surface_elevation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Surface_Elevation\CRMS_Surface_Elevation.csv", encoding='unicode escape')

    dfs = [
        accretion,
        soil_properties,
        # hourly_hydro,
        monthly_hydro,
        marsh_vegetation,
        # forest_vegetation,
        # biomass,
        surface_elevation
    ]
    # Making a common column for dtermining the site name
    for d in range(len(dfs)):
        # if 'Station_ID' in dfs[d].columns:
        #     dfs[d]['Simple site'] = [i[:8] for i in dfs[d]['Station_ID']]
        if 'Station ID' in dfs[d].columns:  # For surface Elevation, soil Properties, marsh vegetation, accretion
            dfs[d]['Simple site'] = [i[:8] for i in dfs[d]['Station ID']]
        if 'CPRA Station ID' in dfs[d].columns:  # For Monthly hydro,
            dfs[d]['Simple site'] = [i[:8] for i in dfs[d]['CPRA Station ID']]

        # Setting the YEARLY dates
        # if 'calendar_year' in dfs[d].columns:
        #     dfs[d]['Year (yyyy)'] = dfs[d]['calendar_year']
        if 'Sample Date (mm/dd/yyyy)' in dfs[d].columns:  # Accretion, soil properties, surface elevation
            dfs[d]['Year (yyyy)'] = organized_iteryears('Sample Date (mm/dd/yyyy)', dfs[d])
        if 'Date (mm/dd/yyyy)' in dfs[d].columns:  # Monthly Hydro,
            dfs[d]['Year (yyyy)'] = organized_iteryears('Date (mm/dd/yyyy)', dfs[d])
        if 'Collection Date (mm/dd/yyyy)' in dfs[d].columns:  # Marsh Veg,
            dfs[d]['Year (yyyy)'] = organized_iteryears('Collection Date (mm/dd/yyyy)', dfs[d])

        # # Set the MONTHLY dates
        # if 'calendar_year' in dfs[d].columns:
        #     dfs[d]['Month (mm)'] = 0  # this means that this data is averaged over a length of years so there is no monthly data
        if 'Sample Date (mm/dd/yyyy)' in dfs[d].columns:  # Accretion, soil properties, surface elevation
            dfs[d]['Month (mm)'] = organized_itermons('Sample Date (mm/dd/yyyy)', dfs[d])
        if 'Date (mm/dd/yyyy)' in dfs[d].columns:  # Monthly Hydro,
            dfs[d]['Month (mm)'] = organized_itermons('Date (mm/dd/yyyy)', dfs[d])
        if 'Collection Date (mm/dd/yyyy)' in dfs[d].columns:  # Marsh Veg,
            dfs[d]['Month (mm)'] = organized_itermons('Collection Date (mm/dd/yyyy)', dfs[d])


        # Add basins: I manually put each site into a basin category, this was done from teh CRMS louisiana website map
        dfs[d]['Basins'] = np.arange(len(dfs[d]['Simple site']))  # this is for appending a basins variable

        if 'Accretion Measurement 1 (mm)' in dfs[d].columns:
            # dfs[d] = add_avgAccretion(dfs[d])
            dfs[d] = add_accretionRate(dfs[d])
        Calcasieu_Sabine = convert_str(
            'CRMS0684, CRMS2189, CRMS0669, CRMS1838, CRMS0665, CRMS2166, CRMS0663, CRMS0662, CRMS2156, CRMS2154, CRMS0697, CRMS0660, CRMS0683, CRMS0661, CRMS2219, CRMS0658, CRMS0693, CRMS0682, CRMS1205, CRMS0651, CRMS0694, CRMS0677, CRMS0680, CRMS1858, CRMS0638, CRMS2334, CRMS0641, CRMS0651, CRMS0635, CRMS0639, CRMS0642, CRMS0647, CRMS6302, CRMS6301, CRMS0685, CRMS0672, CRMS0655, CRMS0687, CRMS0656, CRMS0644, CRMS1743, CRMS1738, CRMS0645, CRMS2418, CRMS0648, CRMS0650, CRMS0691')
        Mermentau = convert_str(
            'CRMS1413, CRMS1409, CRMS0553, CRMS0575, CRMS0622, CRMS0605, CRMS0583, CRMS0587, CRMS0624, CRMS2493, CRMS0614, CRMS0590, CRMS1446, CRMS0584, CRMS0556, CRMS0588, CRMS0615, CRMS0589, CRMS0630, CRMS0593, CRMS0603, CRMS0562, CRMS0604, CRMS0581, CRMS0557, CRMS0554, CRMS0608, CRMS0599, CRMS0595, CRMS0574, CRMS0610, CRMS0626, CRMS0609, CRMS0560, CRMS0600, CRMS1277, CRMS0623, CRMS0567, CRMS0565, CRMS1965, CRMS0576, CRMS1100, CRMS0568, CRMS0570, CRMS1130, CRMS0632, CRMS0572, CRMS0571, CRMS0580, CRMS0633, CRMS0618, CRMS0616, CRMS0619')
        Teche_Vermillion = convert_str(
            'CRMS0536, CRMS0501, CRMS0508, CRMS0507, CRMS2041, CRMS0535, CRMS0552, CRMS1650, CRMS0541, CRMS0511, CRMS0530, CRMS0531, CRMS0532, CRMS0529, CRMS0549, CRMS0527, CRMS0520, CRMS0504, CRMS0514, CRMS0494, CRMS0499, CRMS0498, CRMS0522, CRMS0524, CRMS0523, CRMS0493, CRMS0547, CRMS0550, CRMS0542, CRMS0543, CRMS0513, CRMS0545, CRMS0544, CRMS0517, CRMS5992, CRMS0488, CRMS0551, CRMS0496, CRMS0489, CRMS0490')
        Atchafalaya = convert_str(
            'CRMS6008, CRMS4782, CRMS4779, CRMS4809, CRMS4808, CRMS0479, CRMS4016, CRMS0465, CRMS0464, CRMS6042, CRMS6038, CRMS0482, CRMS2568, CRMS4938, CRMS0461, CRMS4014, CRMS0463, CRMS5003, CRMS6304, CRMS4900')
        Terrebonne = convert_str(
            'CRMS0403, CRMS0324, CRMS5536, CRMS0301, CRMS0365, CRMS0305, CRMS0293, CRMS0309, CRMS5770, CRMS0414, CRMS5035, CRMS2862, CRMS0329, CRMS0399, CRMS0322, CRMS2785, CRMS0290, CRMS0327, CRMS0371, CRMS0326, CRMS0303, CRMS2881, CRMS4045, CRMS0354, CRMS0332, CRMS0377, CRMS0411, CRMS0294, CRMS4455, CRMS0383, CRMS0302, CRMS2887, CRMS0409, CRMS0296, CRMS0398, CRMS0395, CRMS0396, CRMS0381, CRMS0382, CRMS0394, CRMS0376, CRMS0307, CRMS0421, CRMS0311, CRMS0374, CRMS0434, CRMS0345, CRMS0369, CRMS0347, CRMS0390, CRMS0392, CRMS0367, CRMS2939, CRMS0385, CRMS0331, CRMS0315, CRMS0355, CRMS0400, CRMS0341, CRMS3296, CRMS0416, CRMS2825, CRMS0338, CRMS0312, CRMS0386, CRMS0336, CRMS0335, CRMS0387, CRMS0337, CRMS0319, CRMS0318, CRMS0978, CRMS0310, CRMS0397, CRMS0292')
        Barataria = convert_str(
            'CRMS0200, CRMS0197, CRMS0194, CRMS5116, CRMS0217, CRMS5672, CRMS3136, CRMS0206, CRMS0218, CRMS0268, CRMS0192, CRMS0211, CRMS0241, CRMS2991, CRMS3054, CRMS0219, CRMS0273, CRMS3166, CRMS3169, CRMS0189, CRMS0278, CRMS4245, CRMS3985, CRMS0188, CRMS0234, CRMS0185, CRMS0190, CRMS4218, CRMS0261, CRMS4103, CRMS0287, CRMS0248, CRMS0253, CRMS0220, CRMS3565, CRMS6303, CRMS0276, CRMS0225, CRMS0251, CRMS4690, CRMS0232, CRMS3601, CRMS3617, CRMS3590, CRMS0224, CRMS0237, CRMS0226, CRMS0263, CRMS0260, CRMS0258, CRMS3680, CRMS0282, CRMS0209, CRMS4529, CRMS0178, CRMS0175, CRMS0164, CRMS0173, CRMS0176, CRMS0272, CRMS0174, CRMS0179, CRMS0171, CRMS0181, CRMS0172')
        MRD = convert_str(
            'CRMS0163, CRMS2608, CRMS4626, CRMS2634, CRMS0161, CRMS4448, CRMS2627, CRMS0157, CRMS0156, CRMS0162, CRMS0154, CRMS0153, CRMS0159')
        BrentonS = convert_str(
            'CRMS0125, CRMS0128, CRMS0117, CRMS0115, CRMS0114, CRMS0120, CRMS4355, CRMS0131, CRMS0132, CRMS0121, CRMS0146, CRMS0135, CRMS0136, CRMS0147, CRMS0148, CRMS0119, CRMS0129, CRMS0118, CRMS0139, CRMS2614')
        Ponchartrain = convert_str(
            'CRMS0065, CRMS5167, CRMS0008, CRMS0038, CRMS5845, CRMS0039, CRMS0046, CRMS5452, CRMS5267, CRMS0061, CRMS0097, CRMS5373, CRMS0063, CRMS5414, CRMS5255, CRMS0089, CRMS0090, CRMS0058, CRMS0059, CRMS0047, CRMS3913, CRMS0056, CRMS0033, CRMS0034, CRMS0030, CRMS2830, CRMS6299, CRMS6209, CRMS0103, CRMS4094, CRMS2854, CRMS0006, CRMS3667, CRMS4107, CRMS3626, CRMS3650, CRMS0002, CRMS4406, CRMS4407, CRMS3784, CRMS3639, CRMS3641, CRMS3664, CRMS3800, CRMS4548, CRMS4551, CRMS4557, CRMS1024, CRMS0108, CRMS4572, CRMS4596, CRMS0003, CRMS1069')
        Unammed_basin = convert_str('CRMS4110, CRMS0035, CRMS6088, CRMS6090, CRMS0086')

        dfs[d] = add_basins(dfs[d], 'Calcasieu_Sabine', Calcasieu_Sabine)
        dfs[d] = add_basins(dfs[d], 'Mermentau', Mermentau)
        dfs[d] = add_basins(dfs[d], 'Teche_Vermillion', Teche_Vermillion)
        dfs[d] = add_basins(dfs[d], 'Atchafalaya', Atchafalaya)
        dfs[d] = add_basins(dfs[d], 'Terrebonne', Terrebonne)
        dfs[d] = add_basins(dfs[d], 'Barataria', Barataria)
        dfs[d] = add_basins(dfs[d], 'MRD', MRD)
        dfs[d] = add_basins(dfs[d], 'Brenton Sound', BrentonS)
        dfs[d] = add_basins(dfs[d], 'Ponchartrain', Ponchartrain)
        dfs[d] = add_basins(dfs[d], 'Unammed_basin', Unammed_basin)

    return dfs  # Will be a list containing the crms datasets


# The above lines od code above just make the dataframes within the list comparable
# Below I now create functions that manipulate the datasets from the dfs list
def combine_dataframes(dfs):
    ''' this function will take the dataframes and concatenate them (stack them) based on
    their index
    NOTE: Test this again after doing the groupby functions.
    The index and concatenation may be slighly off'''
    i = dfs[0].index.to_flat_index()
    print(i)
    dfs[0].index = i
    full_df = dfs[0]
    for j in range(1, len(dfs)):  # always make sure this is the correct range length.... its confusing me
        # full_df = pd.concat([full_df, dfs[j]], axis=1, ignore_index=False).drop_duplicates()
        idx = dfs[j].index.to_flat_index()
        dfs[j].index = idx
        full_df = pd.concat([full_df, dfs[j]], join='outer', axis=1)
        # full_df = full_df.join(dfs[j], how={'left', 'outer'})
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    return full_df


def average_bysite(dfs):
    '''Below is a df that is craeted by averaging across all years per crms site.
    NOTE: That varaibles constructed by strings are annihilated (due to the .median()command)'''
    for n in range(len(dfs)):
        df = dfs[n].groupby(['Simple site']).median()
        basins = dfs[n].groupby(['Simple site'])['Basins'].agg(pd.Series.mode).to_frame()
        # dfs[n] = pd.concat([df, basins], axis=1)
        # weird thing i decided to do spontaneously, prob can implement better to include more categorical variables
        if 'Community' in dfs[n].columns:
            community = dfs[n].groupby(['Simple site'])['Community'].agg(pd.Series.mode).to_frame()
            # dfs[n] = pd.concat([df, community], axis=1)
            dfs[n] = pd.concat([df, basins, community], axis=1)

        else:
            dfs[n] = pd.concat([df, basins], axis=1)

    full_df = combine_dataframes(dfs)
    return full_df


def average_byyear_bysite_seasonal(dfs):
    '''This will create a dataframe that incorporates data from the 17 years and each season of collection
    It can also take byyear_bymonth,bysite'''
    for n in range(len(dfs)):
        dfs[n]['Season'] = [1 if i > 4 and i <= 10 else 2 for i in dfs[n]['Month (mm)']]
        df = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season']).median()  # Excludes extreme events
        basins = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season'])['Basins'].agg(pd.Series.mode).to_frame()
        # weird thing i decided to do spontaneously, prob can implement better to include more categorical variables
        if 'Community' in dfs[n].columns:
            community = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season'])['Community'].agg(pd.Series.mode).to_frame()
            dfs[n] = pd.concat([df, basins, community], axis=1)
        else:
            dfs[n] = pd.concat([df, basins], axis=1)
    full_df = combine_dataframes(dfs)
    full_df = full_df.reset_index().rename(columns={'level_0':'Simple site', 'level_1':'Year (yyyy)', 'level_2':'Season'})

    return full_df

