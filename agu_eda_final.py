import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\\Etienne\\fall2022\\agu_data\\results\\AGU_dataset.csv", encoding='unicode_escape')

# Main points to prove
# 1. Tidal Amp is important even in a microtidal regime; likely gives us an idea of oceanic influence
# 2. NDVI is only a predictive variable when it is negatively related to accretion; (Show whole plot and fresh v saline)
#    - Show against organic matter % maybe
# 3. Investigate TSS; particularly why it is not important in saline marshes!
# 4. Maybe show salinity with organic matter % variable
# 5. Potentially the histogram of Time inundated variable


# Part 1. Show Tidal Amp is important.
# Show that bulk density and accretion increases with tidal amp
plt.figure()
sns.scatterplot(data=df, x="Tide Amp (ft)", y="Accretion Rate (mm/yr)", hue="Bulk Density (g/cm3)",
                size="Bulk Density (g/cm3)")
plt.show()
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\tides_and_accretion.eps",
            dpi=300, format="eps")

# Part 2. NDVI has a negative influence on accretion in our model
for_part2 = df[(df['Community'] == 'Saline') | (df['Community'] == 'Freshwater') | (df['Community'] == 'Intermediate')]

# Say "There is are varying responses of NDVI depending on marsh type"
plt.figure()
sns.scatterplot(data=for_part2, x="NDVI", y="Accretion Rate (mm/yr)",
                palette=["Red", "Yellow", "Green"], hue="Community", size="Organic Matter (%)")
plt.show()
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\ndvi_and_accretion_communities.eps",
            dpi=300, format="eps")

# Say "Organic Matter (%) in soil may not be well correlated with above ground NDVI"
plt.figure()
sns.scatterplot(data=for_part2, x="NDVI", y="Organic Matter (%)", hue="Community", size="Accretion Rate (mm/yr)",
                palette=["Red", "Yellow", "Green"])
plt.show()
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\ndvi_and_organicMatter_communities.eps",
            dpi=300, format="eps")

plt.figure()
sns.scatterplot(data=for_part2, x="Avg. Flood Depth (ft)", y="Accretion Rate (mm/yr)",
                palette=["Red", "Yellow", "Green"], hue="Community", size="NDVI")
plt.show()
plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\accretion_and_floodDepths_communities.eps",
            dpi=300, format="eps")


