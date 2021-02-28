# Installa TensorFlow

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

inputs = [
    "height3",
    "FG3_attempted_by_minutes_3"
]

data = pd.read_csv('data_nba.csv', usecols=inputs)

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.set_theme(style="white")
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
plt.show()
