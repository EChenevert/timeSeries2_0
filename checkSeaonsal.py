import main
import pandas as pd
import numpy as np

dataframes = main.load()
avgSeasons = main.average_byyear_bysite_seasonal(dataframes)

