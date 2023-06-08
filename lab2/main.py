# Neural Network base
# fast forward backprop
# cascade backprop
# Elman
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import expand_dims
from keras import Input, Model
from keras.layers import concatenate, SimpleRNN
from keras.layers.core import Dense
from keras.models import Sequential


def build_models():
    models = {}

    # 1
    models['1.1x10'] = Sequential([Input(2), Dense(10, activation='relu'), Dense(1)])
    models['1.1x10'].compile(loss='mean_squared_error', metrics=['mean_squared_error'])
    models['1.1x20'] = Sequential([Input(2), Dense(20, activation='relu'), Dense(1)])
    models['1.1x20'].compile(loss='mean_squared_error', metrics=['mean_squared_error'])

    # 2 A
    inputLayer = Input(2)
    hidden_layer1 = Dense(20, activation='relu')(inputLayer)
    outputLayer = Dense(1)(concatenate([inputLayer, hidden_layer1]))

    model2_1_10 = Model(inputLayer, outputLayer)
    model2_1_10.summary()
    model2_1_10.compile(loss='mean_squared_error', metrics=['mean_squared_error'])
    models['2.1x20'] = model2_1_10

    # 2 B
    inputLayer = Input(2)
    hidden_layer1 = Dense(10, activation='relu')(inputLayer)
    hidden_layer2 = Dense(10, activation='relu')(concatenate([inputLayer, hidden_layer1]))
    outputLayer = Dense(1)(concatenate([inputLayer, hidden_layer1, hidden_layer2]))

    model2_2_10 = Model(inputLayer, outputLayer)
    model2_2_10.summary()
    model2_2_10.compile(loss='mean_squared_error', metrics=['mean_squared_error'])
    models['2.2X10'] = model2_2_10

    # 3 A
    models['3.1x15'] = Sequential([
        SimpleRNN(15, activation='relu', input_shape=(1, 2)),
        Dense(1)
    ])
    models['3.1x15'].compile(loss='mean_squared_error', metrics=['mean_squared_error'])

    # 3 B
    models['3.3x5'] = Sequential([
        SimpleRNN(5, activation='relu', input_shape=(1, 2)),
        Dense(5),
        Dense(5),
        Dense(1)
    ])
    models['3.3x5'].compile(loss='mean_squared_error', metrics=['mean_squared_error'])

    return models


def func(x, y):
    return np.sqrt(x ** 2 + y ** 2)


if __name__ == '__main__':
    train_data = np.random.uniform(0, 10, size=(1000, 2))
    label_data = np.array([func(val[0], val[1]) for val in train_data])

    rnn_train_data = np.array(np.reshape(train_data, (np.shape(train_data)[0], 1, np.shape(train_data)[1])).tolist())
    rnn_label_data = np.array(np.reshape(label_data, (np.shape(label_data)[0], 1, 1)).tolist())

    models = build_models()

    plt.figure(figsize=(12, 8))
    for k in models.keys():
        history = models[k].fit(rnn_train_data, rnn_label_data, epochs=200) if k.startswith('3')\
            else models[k].fit(train_data, label_data, epochs=200)
        label = k
        loss = history.history['loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, label=label)
        plt.xlabel("number of epochs")
        plt.ylabel("loss")

    plt.legend()
    plt.show()


