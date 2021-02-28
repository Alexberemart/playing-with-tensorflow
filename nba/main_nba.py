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

number_of_epochs = [
    50,
    60,
    70
]

number_of_neurons = [
    [40, 0],
    [50, 0],
    [60, 0]
]

learning_rate = [
    0.003,
    0.004,
    0.005
]

times = [1, 2, 3, 4, 5]
best_number_of_epochs = 0
best_number_of_neurons = 0
best_learning_rate = 0
best_value = 10000000
best_accuracy = 0
epochs_plot = []
loss_plot = []
neurons_plot = []
learning_rate_plot = []

for lr in learning_rate:

    print("learning_rate " + str(lr))

    for x in number_of_epochs:

        print("epoch " + str(x))

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

                adam = tf.keras.optimizers.Adam(
                    learning_rate=lr
                )

                model.compile(optimizer=adam,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                train = model.fit(x_train, y_train, epochs=x, verbose=0)

                test = model.evaluate(x_test, y_test, verbose=2)

                training_loss += train.history['loss'][-1]
                test_loss += test[0]
                test_accuracy += test[1]

            test_loss_final = test_loss / len(times)
            test_accuracy_final = test_accuracy / len(times)
            if best_value > test_loss_final:
                best_value = test_loss_final
                best_number_of_epochs = x
                best_number_of_neurons = neuron_z[0]
                best_learning_rate = lr
                best_accuracy = test_accuracy_final

            epochs_plot.append(x)
            loss_plot.append(test_loss_final)
            neurons_plot.append(neuron_z[0])
            learning_rate_plot.append(lr)

print("best_number_of_epochs " + str(best_number_of_epochs))
print("best_number_of_neurons " + str(best_number_of_neurons))
print("best_learning_rate " + str(best_learning_rate))
print("best_value " + str(best_value))
print("best_accuracy " + str(best_accuracy))

# https://matplotlib.org/3.3.4/gallery/lines_bars_and_markers/scatter_demo2.html#sphx-glr-gallery-lines-bars-and-markers-scatter-demo2-py

x_plot = [[], [], []]
y_plot = [[], [], []]
c_plot = [[], [], []]
color_1 = []
color_2 = []
color_3 = []
for i in range(len(loss_plot)):
    if neurons_plot[i] == number_of_neurons[0][0]:
        y_plot[0].append(loss_plot[i])
        x_plot[0].append(epochs_plot[i])
        c_plot[0].append(learning_rate_plot[i])
    if neurons_plot[i] == number_of_neurons[1][0]:
        y_plot[1].append(loss_plot[i])
        x_plot[1].append(epochs_plot[i])
        c_plot[1].append(learning_rate_plot[i])
    if neurons_plot[i] == number_of_neurons[2][0]:
        y_plot[2].append(loss_plot[i])
        x_plot[2].append(epochs_plot[i])
        c_plot[2].append(learning_rate_plot[i])

fig, ax = plt.subplots()
ax.scatter(x_plot[0], y_plot[0], c=c_plot[0], s=200, marker='s', alpha=0.5)
ax.scatter(x_plot[1], y_plot[1], c=c_plot[1], s=200, marker='o', alpha=0.5)
ax.scatter(x_plot[2], y_plot[2], c=c_plot[2], s=200, marker='P', alpha=0.5)

ax.set_ylabel('Loss value', fontsize=15)
ax.set_xlabel('Epochs', fontsize=15)
ax.set_title('Loss by hyperparameters')

ax.grid(True)
fig.tight_layout()

plt.show()
