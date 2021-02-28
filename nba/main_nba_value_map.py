import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt

inputs = [
    "height",
    "FG3_attempted_by_minutes_1",
    "position2"
]

data = pd.read_csv('data_nba.csv', usecols=inputs)
data_x = data.to_numpy()

x_plot = []
y_plot = []
c_plot = []
for i in range(len(data)):
    x_plot.append(data_x[i][0])
    y_plot.append(data_x[i][1])
    c_plot.append(data_x[i][2])

fig, ax = plt.subplots()
ax.scatter(x_plot, y_plot, c=c_plot, s=100, marker='o', alpha=0.5)

ax.set_ylabel('FG3 attempted by minutes', fontsize=15)
ax.set_xlabel('Height', fontsize=15)
ax.set_title('Map value')

ax.grid(True)
fig.tight_layout()

plt.show()
