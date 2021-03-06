#/usr/bin/python

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow.keras.backend as K

# reset!
K.clear_session()

# input vectors are 2048 dimensional
inputlayer_conv = keras.Input(shape=(2048,1))

# the first convolutional set
conv1a = keras.layers.Conv1D(filters=16, kernel_size=16, activation='relu')(inputlayer_conv)
conv1b = keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu')(conv1a)
conv1c = keras.layers.Dropout(0.5)(conv1b)
conv1d = keras.layers.MaxPooling1D(pool_size=2)(conv1c)
conv1 = keras.layers.Flatten()(conv1d)
conv_dense = keras.layers.Dense(512, activation='relu')(conv1)
convmodel = keras.Model(inputs=inputlayer_conv, outputs=conv_dense)

# pull in the data via a dense part as well
inputlayer_dense = keras.Input(shape=(2048,))
dense1 = keras.layers.Dense(2048, activation='relu')(inputlayer_dense)
dense2 = keras.layers.Dense(512, activation='relu')(dense1)
densemodel = keras.Model(inputs=inputlayer_dense, outputs=dense2)

# combination step
combined = K.concatenate([convmodel.output, densemodel.output])

combdense = keras.layers.Dense(256, activation='relu')(combined)

model = keras.Model(inputs=[convmodel.input, densemodel.input], outputs=combdense)

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
train_x2 = data['x']
train_y = data['y'].reshape(2000,256,1)

# compile the model and fit it
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit([train_x, train_x2], train_y, batch_size=80, epochs=20, validation_split=0.2)


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


#model.save('linear_42to4.knn')