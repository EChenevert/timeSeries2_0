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

tides = np.asarray(df['Tidal Amplitude (cm)'])
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
ax1.set_xlabel('Tidal Amplitude (cm)')
plt.legend()
plt.show()
fig1.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\tides_accretion_scatterplot.eps",
            dpi=300, format="eps")

# Show that TSS comliments the interpretation that position in tidal frame is related to Suspended Sediment delivery
tss = np.asarray(df['TSS (mg/l)'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
scat2 = ax2.scatter(tss, all_acc, c=bulk, cmap="rocket_r", s=50*10**bulk)
cbar = fig2.colorbar(scat2, ticks=[np.min(bulk), np.max(bulk)])
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
fig2.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\tss_accretion_scatterplot.eps",
            dpi=300, format="eps")



# Part 2. NDVI Looking specifically at difference between Freshwater + Intermediate and Saline Marshes
# Say that there is a clear difference between ndvi in saline marsh and fresh-inter marshes
for_part2 = df[(df['Community'] == 'Saline') | (df['Community'] == 'Freshwater') | (df['Community'] == 'Intermediate')]

sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=for_part2['NDVI'], kde=False,
             hue=for_part2["Community"], palette=["Red", "Orange", "Purple"],
             element="bars", legend=True)
ax.set_title("Distribution of NDVI")
ax.set_xlabel("NDVI")
ax.set_ylabel("Count")
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_histogram.eps",
          dpi=300, format="eps")

# say there is a clear difference in the salinity between saline and fresh-inter marshes
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=for_part2['Soil Porewater Salinity (ppt)'], kde=False,
             hue=for_part2["Community"], palette=["Red", "Orange", "Purple"],
             element="bars", legend=True)
ax.set_title("Distribution of Soil Porewater Salinity (ppt)")
ax.set_xlabel('Soil Porewater Salinity (ppt)')
ax.set_ylabel("Count")
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\salinity_histogram.eps",
          dpi=300, format="eps")

# Show interesting relationship with NDVI and accretion and say that it is related to difference in flooding regimes
flooding = np.asarray(for_part2['Avg. Flood Depth (cm)'])
ndvi = np.asarray(for_part2['NDVI'])
part2_acc = np.asarray(for_part2['Accretion Rate (mm/yr)'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
scat2 = ax2.scatter(ndvi, part2_acc, c=flooding, cmap="rocket_r", s=5*flooding)
cbar = fig2.colorbar(scat2, ticks=[np.min(flooding), np.max(flooding)])
cbar.ax.set_yticklabels([round(np.min(flooding), 2), round(np.max(flooding), 2)])# vertically oriented colorbar
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Avg. Flood Depth (cm)', rotation=270)

ax2.set_ylabel('Accretion Rate (mm/yr)')
ax2.set_xlabel('NDVI')
# plt.legend()
plt.show()
fig2.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_accretion_scatterplot.eps",
            dpi=300, format="eps")


# Say that this is likely due to the salinity flooding brings
salinity = np.asarray(for_part2['Soil Porewater Salinity (ppt)'])

#### Add a plot so that they are on the same scale
fig2, ax2 = plt.subplots(figsize=(8, 6))
scat2 = ax2.scatter(salinity, part2_acc, c=flooding, cmap="rocket_r", s=5*flooding)
cbar = fig2.colorbar(scat2, ticks=[np.min(flooding), np.max(flooding)])
cbar.ax.set_yticklabels([round(np.min(flooding), 2), round(np.max(flooding), 2)])# vertically oriented colorbar
cbar.ax.get_yaxis().labelpad = 20
cbar.set_label('Avg. Flood Depth (cm)', rotation=270)

ax2.set_ylabel('Distribution of Soil Porewater (ppt)')
ax2.set_xlabel('Soil Porewater Salinity (ppt)')
# plt.legend()
plt.show()
fig2.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\salinity_floodDepth_scatterplot.eps",
             dpi=300, format="eps")


##### Showing transitions across marsh gradients

# Tidal: Describes the oceanic influences on the ecosystem and flooding regime (saline versus fresh)
sns.set_theme(style='white', font_scale=1.4)
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(1, 1, 1)
sns.histplot(ax=ax, stat="count", multiple="stack", bins=30,
             x=for_part2['Tidal Amplitude (cm)'], kde=False,
             hue=for_part2["Community"], palette=["Red", "Orange", "Purple"],
             element="bars", legend=True)
ax.set_title("Distribution of Tidal Amplitude (cm)")
ax.set_xlabel('Tidal Amplitude (cm)')
ax.set_ylabel("Count")
f.subplots_adjust(bottom=0.2)
plt.show()
f.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\tides_histogram.eps",
          dpi=300, format="eps")

# Salinity and NDVI: defines the vegetation type and colonization (both are provided above)


## MARSH GRADIENTS FOR ALL DATA POINTS




