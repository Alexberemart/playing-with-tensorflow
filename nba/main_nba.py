# Installa TensorFlow

import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt

inputs = [
    "height3"
]

data = pd.read_csv('nba/data_nba.csv')
point = int((data.shape[0] * 7.5) // 10)
data_x = data[inputs].to_numpy()
data_y = data["position2"].to_numpy()
x_train = data_x[0:point, :].copy()
y_train = data_y[0:point].copy()
x_test = data_x[point:, :].copy()
y_test = data_y[point:].copy()
# data_predict = pd.read_csv('predict.csv')
# x_predict = data_predict[inputs].to_numpy()

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
    256,
    286,
    316,
    346,
    376,
]

number_of_neurons = [
    [256,0],
    [286,0],
    [316,0],
    [346,0],
    [376,0],
]

times = [1, 2, 3, 4, 5]
plot_values = []
best_number_of_epochs = 0
best_number_of_neurons = 0
best_value = 10000000
best_accuracy = 0

for x in number_of_epochs:

    print("epoch " + str(x))
    loss = []

    for neuron_z in number_of_neurons:

        print("neurons " + str(neuron_z[0]))
        training_loss = 0
        test_loss = 0
        test_accuracy = 0

        for y in times:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(neuron_z[0], activation='relu'),
                tf.keras.layers.Dense(neuron_z[0], activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            train = model.fit(x_train, y_train, epochs=x, verbose=0)

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

plt.plot(number_of_epochs, plot_values)
plt.show()
