import tensorflow as tf
import pandas as pd

inputs = [
    "height3",
    "FG3_attempted_by_minutes_3"
]

data = pd.read_csv('nba/data_nba.csv')
point = int((data.shape[0] * 7.5) // 10)
data_x = data[inputs].to_numpy()
data_y = data["position2"].to_numpy()
x_train = data_x[0:point, :].copy()
y_train = data_y[0:point].copy()
x_test = data_x[point:, :].copy()
y_test = data_y[point:].copy()

number_of_epochs = 80
number_of_neurons = 50
learning_rate = 0.002

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(number_of_neurons, activation='relu'),
    tf.keras.layers.Dense(number_of_neurons, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

adam = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train = model.fit(x_train, y_train, epochs=number_of_epochs, verbose=0)

test = model.evaluate(x_test, y_test)

print(test)
