import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r"D:\\Etienne\\fall2022\\agu_data\\results\\AGU_dataset.csv", encoding='unicode_escape')

# Main points to prove
# 1. Tidal Amp is important even in a microtidal regime; likely gives us an idea of oceanic influence
# 2. NDVI is only a predictive variable when it is negatively related to accretion; (Show whole plot and fresh v saline)
#    - Show against organic matter % maybe
# 3. Investigate TSS; particularly why it is not important in saline marshes!
# 4. Maybe show salinity with organic matter % variable
# 5. Potentially the histogram of Time inundated variable


# Using matplot lib to have more control
# # Part 1. Show Tidal Amp is important.
# # Show that bulk density and accretion increases with tidal amp; some sort of oceanic influence on accretion and
# # mineral sediment
plt.rcParams.update({'font.size': 16})

tides = np.asarray(df['Tide Amp (ft)'])
all_acc = np.asarray(df['Accretion Rate (mm/yr)'])
bulk = np.asarray(df['Bulk Density (g/cm3)'])

fig1, ax1 = plt.subplots(figsize=(8, 6))
scat = ax1.scatter(tides, all_acc, c=bulk, cmap="rocket_r", s=50*10**bulk)

cbar = fig1.colorbar(scat, ticks=[np.min(bulk), np.max(bulk)])
cbar.ax.set_yticklabels([round(np.min(bulk), 2), round(np.max(bulk), 2)])# vertically oriented colorbar
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Bulk Density (g/cm3)', rotation=270)

m, b = np.polyfit(tides, all_acc, deg=1)
xseq = np.linspace(0, np.max(tides), num=100)
ax1.plot(xseq, xseq*m + b, "k--", lw=2.5, label="{m}Tide Amp + {b}".format(b=round(b, 2), m=round(m, 2)))
ax1.set_ylabel('Accretion Rate (mm/yr)')
ax1.set_xlabel('Tide Amp (ft)')
plt.legend()
plt.show()

# Show that TSS comliments the interpretation that position in tidal frame is related to Suspended Sediment delivery
tss = np.asarray(df['TSS (mg/l)'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
scat2 = ax2.scatter(tss, all_acc, c=bulk, cmap="rocket_r", s=50*10**bulk)
cbar = fig2.colorbar(scat, ticks=[np.min(bulk), np.max(bulk)])
cbar.ax.set_yticklabels([round(np.min(bulk), 2), round(np.max(bulk), 2)])# vertically oriented colorbar
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Bulk Density (g/cm3)', rotation=270)

m, b = np.polyfit(tss, all_acc, deg=1)
xseq = np.linspace(0, np.max(tss), num=100)
ax2.plot(xseq, xseq*m + b, "k--", lw=2.5, label="{m}TSS + {b}".format(b=round(b, 2), m=round(m, 2)))
ax2.set_ylabel('Accretion Rate (mm/yr)')
ax2.set_xlabel('TSS (mg/l)')
plt.legend()
plt.show()




# Part 2. NDVI Looking specifically at difference between Freshwater + Intermediate and Saline Marshes
# Say that there is a clear difference










# # Part 1. Show Tidal Amp is important.
# # Show that bulk density and accretion increases with tidal amp; some sort of oceanic influence on accretion and
# # mineral sediment
# plt.figure()
# sns.scatterplot(data=df, x="Tide Amp (ft)", y="Accretion Rate (mm/yr)", hue="Bulk Density (g/cm3)",
#                 palette="rocket_r",
#                 size="Bulk Density (g/cm3)")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\tides_and_accretion.eps",
#             dpi=300, format="eps")
#
# # Part 2. NDVI has a negative influence on accretion in our model
# for_part2 = df[(df['Community'] == 'Saline') | (df['Community'] == 'Freshwater') | (df['Community'] == 'Intermediate')]
#
# # Say "There is are varying responses of NDVI depending on marsh type"
# plt.figure()
# sns.scatterplot(data=for_part2, x="NDVI", y="Accretion Rate (mm/yr)",
#                 palette="rocket_r", hue='Avg. Flood Depth (ft)', size='Avg. Flood Depth (ft)')
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\ndvi_and_accretion_communities.eps",
#             dpi=300, format="eps")
#
# plt.figure()
# sns.scatterplot(data=for_part2, x="Soil Porewater Salinity (ppt)", y='Avg. Flood Depth (ft)',
#                 palette="rocket_r", hue="Accretion Rate (mm/yr)", size="Accretion Rate (mm/yr)")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\floodSalAcc.eps",
#             dpi=300, format="eps")
#
# # Say "Organic Matter (%) in soil may not be well correlated with above ground NDVI"
# plt.figure()
# sns.scatterplot(data=for_part2, x="NDVI", y="Organic Matter (%)", hue="Accretion Rate (mm/yr)", size="Accretion Rate (mm/yr)",
#                 palette="rocket_r")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\ndvi_and_organicMatter_communities.eps",
#             dpi=300, format="eps")
#
# plt.figure()
# sns.scatterplot(data=for_part2, x="Avg. Flood Depth (ft)", y="Accretion Rate (mm/yr)",
#                 palette="rocket_r", hue="Bulk Density (g/cm3)", size="Bulk Density (g/cm3)")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\accretion_and_floodDepths_communities.eps",
#             dpi=300, format="eps")
#
# # Part 3. Investigate TSS (mg/l)
# plt.figure()
# sns.scatterplot(data=df, x="TSS (mg/l)", y="Accretion Rate (mm/yr)",
#                 palette="rocket_r", hue="Bulk Density (g/cm3)", size="Bulk Density (g/cm3)")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\accretion_and_TSS.eps",
#             dpi=300, format="eps")
#
# # Part 4. Accretion and Time flooded
# plt.figure()
# sns.scatterplot(data=df, x="Avg. Time Flooded (%)", y="Accretion Rate (mm/yr)",
#                 hue="Bulk Density (g/cm3)", size="Bulk Density (g/cm3)", palette="rocket_r")
# plt.show()
# plt.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\EDA_agu_final\\accretion_and_timeFlooded.eps",
#             dpi=300, format="eps")

