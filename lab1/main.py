# Perceptron
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential

if __name__ == '__main__':
    train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label_data = np.array([[0], [1], [1], [0]])

    model = Sequential([
        Dense(units=16, input_dim=2, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

    model.fit(train_data, label_data, epochs=1000)

    predictions = model.predict(train_data)
    print("Predictions:")
    print(predictions)
