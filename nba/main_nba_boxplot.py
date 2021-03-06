# Installa TensorFlow

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv('nba/data_nba.csv')
data_y = data["position2"].to_numpy()

sns.set_theme(style="whitegrid")
# ax = sns.boxplot(y=data["FG3_attempted_by_minutes_1"], x=data_y)
# ax = sns.boxplot(y=data["height"], x=data_y)
ax = sns.boxplot(y=data["blocks_by_minutes_1"], x=data_y)
plt.show()
