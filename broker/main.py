# Installa TensorFlow

import tensorflow as tf
import pandas as pd

# import numpy as np
# import seaborn as sns

import matplotlib.pyplot as plt

inputs = [
    "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday", "is_january", "is_february", "is_march",
    "is_april", "is_may", "is_june", "is_july", "is_august", "is_september", "is_october", "is_november", "is_december",
    "tlf_vol_LD_gap_1", "tlf_vol_LD_gap_2", "tlf_vol_LD_gap_3", "tlf_vol_LD_gap_4", "tlf_ydy_gap_1", "tlf_ydy_gap_2",
    "tlf_ydy_gap_3", "tlf_ydy_gap_4", "i35_ydy_gap_1", "i35_ydy_gap_2", "i35_ydy_gap_3", "i35_ydy_gap_4",
    "tlf_L7D_gap_1", "tlf_L7D_gap_2", "tlf_L7D_gap_3", "tlf_L7D_gap_4", "i35_L7D_gap_1", "i35_L7D_gap_2",
    "i35_L7D_gap_3", "i35_L7D_gap_4", "i35_LD_gap_1", "i35_LD_gap_2", "i35_LD_gap_3", "i35_LD_gap_4", "i35_LD_gap_5",
    "i35_LD_gap_6", "i35_LD_gap_7", "tlf_LD_gap_1", "tlf_LD_gap_2", "tlf_LD_gap_3", "tlf_LD_gap_4"
]

data = pd.read_csv('data.csv')
point = int((data.shape[0] * 7.5) // 10)
data_x = data[inputs].to_numpy()
data_y = data["rep_result"].to_numpy()
x_train = data_x[0:point, :].copy()
y_train = data_y[0:point].copy()
x_test = data_x[point:, :].copy()
y_test = data_y[point:].copy()
data_predict = pd.read_csv('predict.csv')
x_predict = data_predict[inputs].to_numpy()

# corr = data_train.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.set_theme(style="white")
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5});
# plt.show()

number_of_epochs = [
    1,
    2,
    4,
    8,
    16
]

number_of_neurons = [
    [4, 2],
    [8, 3],
    [16, 4]
]

times = [1, 2, 3, 4, 5]
plot_values = []
best_number_of_epochs = 0
best_number_of_neurons = 0
best_value = 10000000
best_accuracy = 0

for x in number_of_epochs:

    loss = []

    for neuron_z in number_of_neurons:

        training_loss = 0
        test_loss = 0
        test_accuracy = 0

        for y in times:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(neuron_z[0], activation='relu'),
                tf.keras.layers.Dense(neuron_z[0], activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            train = model.fit(x_train, y_train, epochs=x)

            test = model.evaluate(x_test, y_test, verbose=2)
            # print(y_test)
            # print(test)

            # predicted = model.predict(x_predict)
            # print(predicted)

            training_loss += train.history['loss'][-1]
            test_loss += test[0]
            test_accuracy += test[1]

        test_loss_final = test_loss / len(times)
        test_accuracy_final = test_accuracy / len(times)
        if best_value > test_loss_final:
            best_value = test_loss_final
            best_number_of_epochs = x
            best_number_of_neurons = neuron_z[0]
            best_accuracy = test_accuracy_final

        loss.append(training_loss / len(times) + neuron_z[1])
        loss.append(test_loss_final + neuron_z[1])

    plot_values.append(loss)

print("best_number_of_epochs " + str(best_number_of_epochs))
print("best_number_of_neurons " + str(best_number_of_neurons))
print("best_value " + str(best_value))
print("best_accuracy " + str(best_accuracy))

# plt.plot(number_of_epochs, plot_values)
# plt.show()
