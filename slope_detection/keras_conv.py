#/usr/bin/python

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow.keras.backend as K

#
# This trains the neural net to learn the derivative given the window profile
#


# reset!
K.clear_session()

# input vectors are 2048 dimensional
inputlayer = keras.Input(shape=(2048,1))

# the first convolutional set
conv1a = keras.layers.Conv1D(filters=16, kernel_size=16, activation='relu')(inputlayer)
conv1b = keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu')(conv1a)
conv1c = keras.layers.Dropout(0.1)(conv1b)
conv1d = keras.layers.MaxPooling1D(pool_size=2)(conv1c)
conv1 = keras.layers.Flatten()(conv1d)
dense1 = keras.layers.Dense(512, activation='relu')(conv1)
dense2 = keras.layers.Dense(256, activation='linear')(dense1)
model = keras.Model(inputs=inputlayer, outputs=dense2)

# the output layer
#outputlayer = keras.layers.Dense(256, activation='relu')(conv1)
# create the keras model

#model = keras.Model(inputs=inputlayer, outputs=outputlayer)


# oof

#model = keras.models.Sequential()
#model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='same', input_shape=(2048,1)))
#model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
#model.add(keras.layers.Dropout(0.1))
#model.add(keras.layers.MaxPooling1D(pool_size=4))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(1024, activation='relu'))
#model.add(keras.layers.Dense(512, activation='relu'))
#model.add(keras.layers.Dense(256, activation='linear'))


# load the data
data = np.load('windows_2048to256.npz', 'r')

train_x = data['x'].reshape(2000,2048,1)
train_y = data['y'].reshape(2000,256,1)

train_x = np.ones(shape=train_x.shape) - train_x
train_y = (-1) * train_y / np.amax(train_y)

# compile the model and fit it
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_x, train_y, batch_size=80, epochs=100, validation_split=0.2)


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

a = train_x[0].reshape(1,2048,1)

plt.plot(np.linspace(0,256,256), model.predict(a).flatten)
plt.plot(np.linspace(0,256,256), train_y[0])


model.save('windows_conv_2048to256.knn')