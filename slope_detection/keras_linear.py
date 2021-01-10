#/usr/bin/python

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow.keras.backend as K

# reset!
K.clear_session()

# input vectors are 6 dimensional
inputlayer = keras.Input(shape=(42,))

# next, a dense layer
dense1 = keras.layers.Dense(84, activation='relu')(inputlayer)

# possibly more
dense2 = keras.layers.Dense(42, activation='relu')(dense1)
#dense3 = keras.layers.Dense(21, activation='relu')(dense2)

# the output layer
outputlayer = keras.layers.Dense(4, activation='linear')(dense2)



# create the keras model

model = keras.Model(inputs=inputlayer, outputs=outputlayer)

model.summary()

# load the data
data = np.load('linear_42to4.npz', 'r')

train_x = data['x']
train_y = data['y']

# compile the model and fit it
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_x, train_y, batch_size=20, epochs=100, validation_split=0.2)


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


#print(model.predict(np.array([[1,2,3,4,5,6]])))

a = np.array([[1] + [0 for x in range(41)]])
b = train_x[0].reshape((1,42,))
m = np.load('matrix.npz', 'r')['M'][:42, :4]

print(a @ m)
print(model.predict(a))
print(b @ m)
print(model.predict(b))

model.save('linear_42to4.knn')