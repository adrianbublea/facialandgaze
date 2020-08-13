# coding:utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from input_data import load2d

def nn_model():    
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30))

    return model


def plot_loss(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(loss, linewidth = 2, label = 'train')
    plt.plot(val_loss, linewidth = 2, label = 'valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def check_test(model):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap = 'gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker = 'x', s = 10)

    X, _ = load2d(test = True)
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    plt.show()

def main():
    PRETRAIN = False

    X, y = load2d()

    model = nn_model()
    if PRETRAIN:
        model.load('my_model.h5')
        
    sgd = SGD(lr = 0.01, momentum = 0.9, nesterov=True)
    model.compile(loss = 'mse', optimizer = sgd, metrics = ['accuracy'])

    hist = model.fit(X, y, epochs = 10000, batch_size = 30, verbose = 1, validation_split = 0.2)

    model.save('my_model.h5')
    #model.save('my_model2.h5')
    np.savetxt('my_nn_model_loss.csv', hist.history['loss'])
    np.savetxt('my_nn_model_val_loss.csv', hist.history['val_loss'])

    plot_loss(hist)
    check_test(model)

if __name__ == '__main__':
    main()
