#/usr/bin/python

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow.keras.backend as K

# reset!
K.clear_session()

# input vectors are 6 dimensional
inputlayer = keras.Input(shape=(2048,))

# next, a dense layer
dense1 = keras.layers.Dense(2048, activation='elu')(inputlayer)

# possibly more
dense2 = keras.layers.Dense(1024, activation='relu')(dense1)
dense3 = keras.layers.Dense(512, activation='relu')(dense2)

# the output layer
outputlayer = keras.layers.Dense(256, activation='linear')(dense3)



# create the keras model

model = keras.Model(inputs=inputlayer, outputs=outputlayer)

model.summary()

# load the data
data = np.load('windows_2048to256.npz', 'r')

train_x = data['x']
train_y = data['y']

train_x = np.ones(shape=train_x.shape) - train_x

train_y = (-1) * train_y / np.amax(train_y)

# compile the model and fit it
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_x, train_y, batch_size=64, epochs=10, validation_split=0.2)


# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#model.save('windows_deriv_2048to256.knn')